"""Compare Prototype, Full RLS and fixed random linear AFE-RLS on frozen V1 features.
用相同的冻结 MSWC V1 128D 特征比较原型、完整 RLS 与随机线性扩展 RLS。

AFE is a component experiment inspired by AnalyticKWS, NOT a full reproduction.
AFE 是借鉴 AnalyticKWS 的组件实验，并非整篇论文的严格复现。
Default: h = z @ R, R_ij ~ N(0, 1 / output_dim), no bias/ReLU/extra normalization.
默认：h = z @ R，矩阵元素方差为 1/输出维度，无偏置、ReLU 或额外归一化。
This scaling preserves squared feature norm in expectation; it is our explicit
experimental convention, not a claim about the paper's unspecified variance.
该缩放在期望上保持特征平方范数，是本实验明确选定的约定，不声称是论文原始方差。

The original two baselines and original default base_shots behavior are preserved.
保留原有两条基线和 base_shots 的默认行为；固定初始类10条时请写 --base-shots 10。
Use --no-afe to run only the original baselines. Separate lambdas can be selected
on development data; the script never automatically tunes on held-out test data.
用 --no-afe 仅跑原始基线。两条 RLS 可分别设置正则化；脚本不自动用最终测试集调参。

History arrays and BRR are desktop verification only, not required MCU state.
历史样本数组及 BRR 仅供电脑校验，并非部署到 MCU 时必须保留的学习状态。

从项目根目录运行，并将本文件放在 evaluate_mswc_20_unseen_10shot_128d.py 旁。
Run from the project root with that existing evaluator next to this file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import hashlib
import warnings

import numpy as np


BASE_COUNT = 6
FEATURE_DIM = 128
SUPPORT_POOL = 40
QUERIES_PER_CLASS = 50
DEFAULT_SEEDS = (42, 123, 2026, 7, 99)


class FullRLS:
    """Dense FP32 RLS with one-hot targets and no forgetting.
    密集 FP32 RLS，采用独热目标，不使用遗忘因子。
    """

    def __init__(self, dimension, classes, ridge_lambda):
        # P starts at (lambda I)^-1. Each column of W scores one class.
        # P 从 (lambda I)^-1 开始，W 的每一列对应一个类别的分数。
        self.p = np.eye(dimension, dtype=np.float32) / np.float32(ridge_lambda)
        self.w = np.zeros((dimension, classes), dtype=np.float32)

    def add(self, vector, label, weight):
        # Scaling both x and y by sqrt(weight) gives weighted least squares.
        # 将输入与目标同时乘 sqrt(weight)，得到对应的加权最小二乘。
        root_weight = np.sqrt(np.float32(weight))
        x = vector * root_weight
        px = self.p @ x
        denominator = np.float32(1) + x @ px
        # A positive denominator follows from positive-definite P in exact arithmetic.
        # 理论上正定 P 保证分母为正；异常数值立即报错，避免继续输出误导性结果。
        if not np.isfinite(denominator) or denominator <= 0:
            raise FloatingPointError("Invalid RLS denominator; check feature scale/lambda")
        gain = px / denominator
        residual = -(x @ self.w)
        residual[label] += root_weight
        self.w += np.outer(gain, residual)
        self.p -= np.outer(px, px) / denominator


def batch_ridge(x, labels, weights, classes, ridge_lambda):
    """Independent FP64 reference for the same objective and registration history.
    用独立的 FP64 Batch Ridge 求解相同目标函数和相同注册历史。
    """
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    targets = np.eye(classes, dtype=np.float64)[labels]
    xt_weighted = x.T * weights
    return np.linalg.solve(
        xt_weighted @ x + ridge_lambda * np.eye(x.shape[1]),
        xt_weighted @ targets,
    )


def _evaluate_feature_space(words, support, queries, seed, shots, base_shots,
                     new_weight, ridge_lambda):
    """Build one shared enrollment plan and evaluate every cumulative session.
    建立一份共享注册计划，按顺序评价每个累计增量阶段。
    """
    base_words = words[:BASE_COUNT]
    novel_words = words[BASE_COUNT:].copy()
    random.Random(seed + 100_000).shuffle(novel_words)
    order = base_words + novel_words
    rng = random.Random(seed)

    # Sample before training. Sampling in manifest order keeps registrations
    # unchanged when the order of insertion differs. Smaller shot counts are
    # prefixes of the same permutation.
    # 提前按清单顺序抽样，新增类顺序变化时不会重新指定各词样本。
    # 较少 shot 的注册样本是相同随机排列的前缀。
    selected = {
        word: rng.sample(range(len(support[word])), len(support[word]))[
            :base_shots if word in base_words else shots
        ]
        for word in words
    }
    rls = FullRLS(support[order[0]].shape[1], len(order), ridge_lambda)
    centers, history_x, history_y, history_weights = [], [], [], []
    rows, base_accuracy = [], {}
    for label, word in enumerate(order):
        registration = support[word][selected[word]]
        centers.append(registration.mean(axis=0, dtype=np.float32))
        weight = 1.0 if label < BASE_COUNT else new_weight
        for vector in registration:
            rls.add(vector, label, weight)
            history_x.append(vector)
            history_y.append(label)
            history_weights.append(weight)

        # All six base classes are initialized before reporting session 0.
        # 先注册完六个基础词，再评价第 0 阶段。
        active = label + 1
        if active < BASE_COUNT:
            continue
        active_words = order[:active]
        blocks = [queries[w] for w in active_words]
        x_query = np.concatenate(blocks)
        lengths = [len(block) for block in blocks]
        y_query = np.repeat(np.arange(active), lengths)

        # Full RLS keeps zero-target history for future classes through P.
        # Predict only among registered classes. Compare against BRR at every stage.
        # P 会保留先前样本对后续类别的影响。预测只看已注册的类，每阶段和 BRR 核对。
        reference = batch_ridge(history_x, history_y, history_weights, active, ridge_lambda)
        online = rls.w[:, :active]
        difference = online - reference
        relative_error = float(np.linalg.norm(difference) /
                               max(np.linalg.norm(reference), np.finfo(float).eps))
        if not np.isfinite(relative_error):
            raise FloatingPointError(f"Non-finite RLS state at seed={seed}, classes={active}")
        rls_pred = (x_query @ online).argmax(axis=1)
        brr_pred = (x_query.astype(np.float64) @ reference).argmax(axis=1)
        mismatches = int(np.count_nonzero(rls_pred != brr_pred))

        # Squared Euclidean distance to the FP32 mean of each class.
        # 用平方欧氏距离比较查询向量与各类 FP32 均值原型。
        centroid_matrix = np.stack(centers)
        displacement = x_query[:, None, :] - centroid_matrix[None, :, :]
        proto_pred = np.einsum("ncd,ncd->nc", displacement, displacement).argmin(axis=1)
        end_indices = np.cumsum(lengths)[:-1]
        for method, pred in (("Prototype", proto_pred), ("Full RLS", rls_pred)):
            correct_blocks = np.split(pred == y_query, end_indices)
            per_class = np.array([part.mean() * 100 for part in correct_blocks])
            old = float(per_class[:BASE_COUNT].mean())
            new = float(per_class[BASE_COUNT:].mean()) if active > BASE_COUNT else None
            base_accuracy.setdefault(method, old)
            rows.append(dict(
                seed=seed, base_shots=base_shots, new_shots=shots,
                session=active - BASE_COUNT, active_classes=active,
                added_word=word if active > BASE_COUNT else "", method=method,
                old_pct=old, new_pct=new, macro_pct=float(per_class.mean()),
                # new_pct averages ALL added classes; latest_new_pct is only this session's word.
                # new_pct 是所有已加入新类的平均准确率；latest_new_pct 仅对应本阶段刚加入的词。
                latest_new_pct=float(per_class[-1]) if active > BASE_COUNT else None,
                old_drop_pp=base_accuracy[method] - old,
                rls_brr_rel_weight_error=relative_error if method == "Full RLS" else None,
                rls_brr_prediction_mismatches=mismatches if method == "Full RLS" else None,
            ))
    return rows, order, selected


def make_expansion(input_dim, output_dim, seed, scale="norm-preserving"):
    """Create one fixed FP32 Gaussian matrix, independent of enrollment randomness.
    创建独立于注册抽样随机数的固定 FP32 高斯矩阵。

    norm-preserving: std=1/sqrt(output_dim), E[||zR||²]=||z||².
    norm-preserving：标准差为 1/sqrt(输出维度)，期望上保持平方范数。
    standard: std=1; useful for an explicitly labelled scale ablation.
    standard：标准差为1，可用于明确标注的尺度消融实验。
    Neither option adds information or makes the mapping nonlinear.
    两种设置都不会凭空增加信息，也不会把线性映射变为非线性。
    """
    if output_dim <= input_dim:
        raise ValueError("AFE output dimension must exceed input dimension")
    if scale not in ("norm-preserving", "standard"):
        raise ValueError("Unknown AFE scale")
    rng = np.random.default_rng(seed)
    std = 1.0 / np.sqrt(output_dim) if scale == "norm-preserving" else 1.0
    return (rng.standard_normal((input_dim, output_dim)) * std).astype(np.float32)


def expand_features(groups, matrix):
    """Transform every word using the SAME matrix; never mutate raw embeddings.
    所有词共用同一矩阵变换，绝不原地修改原始 embedding。
    A row is one recording, so (N,d) @ (d,D) produces (N,D).
    每一行代表一条录音，因此 (N,d) @ (d,D) 得到 (N,D)。
    """
    result = {word: np.asarray(values @ matrix, dtype=np.float32)
              for word, values in groups.items()}
    if not all(np.isfinite(values).all() for values in result.values()):
        raise FloatingPointError("Non-finite expanded features")
    return result


def evaluate_episode(words, support, queries, seed, shots, base_shots,
                     new_weight, ridge_lambda, expansion=None, afe_ridge_lambda=None):
    """Keep raw baselines unchanged, then evaluate AFE in an independent state.
    保持原始基线不变，再使用独立状态评价 AFE 分支。

    Both calls use exactly the same deterministic enrollment plan and ordering.
    两次调用使用完全相同的确定性注册计划和类别顺序，并显式检查其一致性。
    The expanded-space Prototype is discarded: it is NOT the raw Prototype baseline.
    扩展空间内辅助计算的 Prototype 被丢弃，不能冒充原始 Prototype 基线。
    Without expansion this function retains the original two-method interface.
    不传入 expansion 时，保留旧版两方法接口，便于回归校验。
    """
    raw_rows, order, selected = _evaluate_feature_space(
        words, support, queries, seed, shots, base_shots, new_weight, ridge_lambda)
    dimension = support[words[0]].shape[1]
    for row in raw_rows:
        is_rls = row["method"] == "Full RLS"
        row.update(feature_dim=dimension, ridge_lambda=ridge_lambda if is_rls else None,
                   p_bytes=4 * dimension**2 if is_rls else 0,
                   classifier_bytes=4 * dimension * (len(words) if is_rls else row["active_classes"]),
                   projection_bytes=0)
    if expansion is not None:
        afe_lambda = ridge_lambda if afe_ridge_lambda is None else afe_ridge_lambda
        expanded_rows, other_order, other_selected = _evaluate_feature_space(
            words, expand_features(support, expansion), expand_features(queries, expansion),
            seed, shots, base_shots, new_weight, afe_lambda)
        if order != other_order or selected != other_selected:
            raise RuntimeError("AFE and baseline enrollment plans differ")
        dimension = expansion.shape[1]
        for row in expanded_rows:
            if row["method"] != "Full RLS":
                continue
            row.update(method=f"AFE-RLS-{dimension}", feature_dim=dimension,
                       ridge_lambda=afe_lambda, p_bytes=4 * dimension**2,
                       classifier_bytes=4 * dimension * len(words),
                       projection_bytes=int(expansion.nbytes))
            raw_rows.append(row)
    # Bytes cover FP32 model arrays only, not history, temporary arrays, audio or CNN.
    # 字节数只覆盖 FP32 模型矩阵，不包括历史样本、临时数组、音频或 CNN；不是峰值 RAM。
    for row in raw_rows:
        row["model_array_bytes"] = row["p_bytes"] + row["classifier_bytes"] + row["projection_bytes"]
    raw_rows.sort(key=lambda row: (row["session"],
                  {"Prototype": 0, "Full RLS": 1}.get(row["method"], 2)))
    return raw_rows, order, selected


def save_summary(path, rows):
    """Summarize each session across enrollment/order seeds, not across sessions.
    对每个阶段在注册/顺序 seed 间汇总，不把不同增量阶段混作重复实验。
    SD is sample standard deviation (ddof=1); one seed has no estimable SD.
    SD 为样本标准差（ddof=1）；只有一个 seed 时留空，不伪称零不确定性。
    AFE seed is fixed within this run, so SD does not measure projection randomness.
    一次运行内 AFE seed 固定，因此 SD 不衡量随机投影带来的不确定性。
    """
    result = []
    keys = dict.fromkeys((r["session"], r["method"]) for r in rows)
    for session, method in keys:
        group = [r for r in rows if (r["session"], r["method"]) == (session, method)]
        record = dict(session=session, active_classes=group[0]["active_classes"],
                      method=method, seeds=len(group))
        for key in ("old_pct", "new_pct", "latest_new_pct", "macro_pct", "old_drop_pp"):
            values = [r[key] for r in group if r[key] is not None]
            record[key + "_mean"] = float(np.mean(values)) if values else None
            record[key + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result.append(record)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)


def load_protocol(base):
    """Reuse the fixed MSWC validation files and verify vocabulary and speakers.
    复用固定 MSWC 验证文件，并检查预训练词表和说话人隔离。
    """
    manifest = json.loads(base.VOCAB.read_text(encoding="utf-8-sig"))
    words = [word.strip().lower() for word in manifest["development_words"]]
    final_words = {word.strip().lower() for word in manifest["final_heldout_words"]}
    with (base.ROOT / "full_vocab_1p5s/class_map.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        trained_words = {row["word"].strip().lower() for row in csv.DictReader(handle)}
    if len(words) != 20 or len(set(words)) != 20 or set(words) & (final_words | trained_words):
        raise RuntimeError("The development vocabulary has duplicates or overlaps protected sets")
    support = base.read_manifest("novel_dev_support_pool.csv", SUPPORT_POOL, words, "train")
    queries = base.read_manifest("novel_dev_query.csv", QUERIES_PER_CLASS, words, "dev")
    support_files, support_speakers = map(set, zip(*(r for w in words for r in support[w])))
    query_files, query_speakers = map(set, zip(*(r for w in words for r in queries[w])))
    if "" in support_speakers | query_speakers:
        raise RuntimeError("Missing speaker IDs in MSWC development manifests")
    if support_files & query_files or support_speakers & query_speakers:
        raise RuntimeError("Enrollment and validation queries overlap in recordings or speakers")
    return words, support, queries


def self_test():
    """Check the complete session loop without audio files or PyTorch.
    使用合成特征测试完整增量循环，无需音频文件或 PyTorch。
    """
    rng = np.random.default_rng(12)
    words = [f"word{i}" for i in range(9)]
    centers = np.eye(len(words), dtype=np.float32)
    support = {
        word: (centers[i] + rng.normal(0, 0.025, (40, len(words)))).astype(np.float32)
        for i, word in enumerate(words)
    }
    queries = {word: values[:8] + np.float32(0.005) for word, values in support.items()}
    for base_shots, weight in ((10, 1.0), (15, 2.5)):
        rows, order, selected = evaluate_episode(words, support, queries, 42, 10,
                                                  base_shots, weight, 1.0)
        assert order[:BASE_COUNT] == words[:BASE_COUNT]
        assert len(selected[words[0]]) == base_shots
        assert len(rows) == 2 * (len(words) - BASE_COUNT + 1)
        assert rows[0]["new_pct"] is None
        assert rows[-2]["new_pct"] == rows[-1]["new_pct"] == 100.0
        for row in rows[1::2]:
            assert row["rls_brr_rel_weight_error"] < 1e-5
            assert row["rls_brr_prediction_mismatches"] == 0
    # Independent synthetic queries avoid relying on the original overlapping smoke data.
    # 使用独立生成的查询样本，额外测试不依赖旧版自测中重叠的合成数据。
    queries = {word: (centers[i] + rng.normal(0, 0.025, (12, len(words)))).astype(np.float32)
               for i, word in enumerate(words)}
    matrix = make_expansion(len(words), 24, 31415)
    assert np.array_equal(matrix, make_expansion(len(words), 24, 31415))
    saved_support = {w: x.copy() for w, x in support.items()}
    for shots in (1, 5, 10):
        for weight in (1.0, 2.5):
            baseline, order, selected = evaluate_episode(words, support, queries, 42, shots, 10, weight, 1.0)
            extended, order2, selected2 = evaluate_episode(
                words, support, queries, 42, shots, 10, weight, 1.0, matrix, 0.5)
            assert order == order2 and selected == selected2
            assert baseline == [r for r in extended if not r["method"].startswith("AFE")]
            assert len(extended) == 3 * (len(words) - BASE_COUNT + 1)
            for row in extended:
                if row["rls_brr_rel_weight_error"] is not None:
                    assert row["rls_brr_rel_weight_error"] < 1e-4
                    assert row["rls_brr_prediction_mismatches"] == 0
            assert all(np.array_equal(support[w], saved_support[w]) for w in words)
    print("SELF-TEST PASS | raw baseline parity | AFE shared enrollment | 1/5/10-shot | weighted BRR parity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=10)
    parser.add_argument("--base-shots", type=int, help="Default: same as --shots")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--ridge-lambda", type=float, default=1.0)
    parser.add_argument("--new-weight", type=float, default=1.0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    # A separate AFE seed never changes support sampling or class order.
    # 独立的 AFE seed 不影响注册录音抽样和类别加入顺序。
    parser.add_argument("--afe-dim", type=int, default=256)
    parser.add_argument("--afe-seed", type=int, default=31415)
    parser.add_argument("--afe-scale", choices=("norm-preserving", "standard"), default="norm-preserving")
    parser.add_argument("--afe-ridge-lambda", type=float, help="Default: same as --ridge-lambda")
    parser.add_argument("--no-afe", action="store_true", help="Run original two baselines only")
    parser.add_argument("--parity-tol", type=float, default=1e-3,
                        help="Relative weight error warning threshold; not an accuracy tolerance")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    afe_lambda = args.ridge_lambda if args.afe_ridge_lambda is None else args.afe_ridge_lambda
    if not args.no_afe and (args.afe_dim <= FEATURE_DIM or args.afe_seed < 0):
        parser.error("AFE dimension must exceed 128 and AFE seed must be nonnegative")
    if not all(np.isfinite(v) and v > 0 for v in (afe_lambda, args.parity_tol)):
        parser.error("AFE lambda and parity tolerance must be finite and positive")
    base_shots = args.shots if args.base_shots is None else args.base_shots
    if not 1 <= args.shots <= SUPPORT_POOL or not 1 <= base_shots <= SUPPORT_POOL:
        parser.error("base and new shots must each be between 1 and 40")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be distinct")
    if not all(np.isfinite(value) and value > 0 for value in
               (args.ridge_lambda, args.new_weight)):
        parser.error("ridge lambda and new-class weight must be finite and positive")

    # Keep the synthetic self-test independent of the project dependencies.
    # 合成数据自测无需安装原项目的音频或深度学习依赖。
    import evaluate_mswc_20_unseen_10shot_128d as base
    import torch

    words, support_plan, query_plan = load_protocol(base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = base.MODEL_LARGE
    print(f"Protocol PASS | model={checkpoint} | device={device}", flush=True)
    print(f"Base words: {', '.join(words[:BASE_COUNT])} | train speakers disjoint from dev",
          flush=True)
    # Use the original V1 audio, Log-Mel and caching path, then cast once to FP32.
    # 沿用 V1 的音频、Log-Mel 和缓存流程，然后一次性转为 FP32。
    support = base.extract_all(support_plan, words, checkpoint, 24000, device, 64)
    queries = base.extract_all(query_plan, words, checkpoint, 24000, device, 64)
    for groups in (support, queries):
        for word, features in groups.items():
            if features.ndim != 2 or features.shape[1] != FEATURE_DIM or not np.isfinite(features).all():
                raise RuntimeError(f"Expected finite V1 128D embeddings for {word}")
            groups[word] = features.astype(np.float32, copy=False)

    # Generate once, outside all episodes: the same matrix serves old/new/query data.
    # 在全部 episode 外只生成一次；旧类、新类、查询数据统一使用该矩阵。
    expansion = None if args.no_afe else make_expansion(
        FEATURE_DIM, args.afe_dim, args.afe_seed, args.afe_scale)
    rows, plans = [], {}
    for seed in args.seeds:
        episode, order, selected = evaluate_episode(
            words, support, queries, seed, args.shots, base_shots,
            args.new_weight, args.ridge_lambda, expansion, afe_lambda,
        )
        rows.extend(episode)
        plans[str(seed)] = {
            "class_order": order,
            "registration_files": {
                word: [support_plan[word][i][0] for i in selected[word]] for word in order
            },
        }
        for row in episode:
            new = "N/A" if row["new_pct"] is None else f'{row["new_pct"]:.2f}%'
            print(f'{seed:4d} | classes={row["active_classes"]:2d} | {row["method"]:9s} | '
                  f'old={row["old_pct"]:.2f}% new={new} macro={row["macro_pct"]:.2f}% '
                  f'old drop={row["old_drop_pp"]:.2f} pp', flush=True)

    out = args.out or Path(f"results/v1_prototype_vs_rls_base{base_shots}_new{args.shots}"
                           f"_lambda{args.ridge_lambda:g}_weight{args.new_weight:g}"
                           + ("_noafe" if args.no_afe else
                              f"_afe{args.afe_dim}_seed{args.afe_seed}_{args.afe_scale}_lambda{afe_lambda:g}"))
    # Do not silently erase a previous experiment when reusing --out.
    # 重用 --out 时不静默覆盖旧实验，避免弄丢结果或混淆不同设置。
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose another --out: {out}")
    out.mkdir(parents=True, exist_ok=True)
    save_summary(out / "summary.csv", rows)
    if expansion is not None:
        # Saving the actual matrix removes dependence on future RNG implementation details.
        # 保存实际矩阵，避免未来随机数实现变化影响复现。
        np.save(out / "afe_projection.npy", expansion, allow_pickle=False)
    base.save_csv(out / "sessions.csv", rows)
    (out / "registration_plan.json").write_text(json.dumps(dict(
        checkpoint=str(checkpoint), ridge_lambda=args.ridge_lambda, new_weight=args.new_weight,
        base_shots=base_shots, new_shots=args.shots, seeds=plans,
        numpy_version=np.__version__, protocol="development; speaker-disjoint; final heldout untouched",
        afe_enabled=not args.no_afe, afe_dim=args.afe_dim if not args.no_afe else None,
        afe_seed=args.afe_seed if not args.no_afe else None,
        afe_scale=args.afe_scale if not args.no_afe else None,
        afe_ridge_lambda=afe_lambda if not args.no_afe else None,
        afe_std=(1.0 / np.sqrt(args.afe_dim) if args.afe_scale == "norm-preserving" else 1.0)
                if not args.no_afe else None,
        afe_sha256=hashlib.sha256(expansion.tobytes()).hexdigest() if expansion is not None else None,
        afe_activation="none", extra_normalization="none; upstream extractor behavior preserved",
        metrics_note="old=initial six; new=all added; latest_new=last added; candidates=all seen",
        memory_note="FP32 arrays only; W preallocates final class count; excludes CNN/audio/history/temporaries",
        parity_warning_tolerance=args.parity_tol,
    ), indent=2), encoding="utf-8")
    # Numerical warnings are visible, but small FP32 tie-related mismatches are not
    # incorrectly equated with a broken algorithm. Always inspect the recorded errors.
    # 数值问题明确警告；FP32 近似并列分数造成的预测差异不直接等同于算法错误。
    # 若警告出现，应先检查相对权重误差、特征尺度和正则化，再解释准确率。
    for method in dict.fromkeys(r["method"] for r in rows if r["rls_brr_rel_weight_error"] is not None):
        selected_rows = [r for r in rows if r["method"] == method]
        worst = max(r["rls_brr_rel_weight_error"] for r in selected_rows)
        mismatches = sum(r["rls_brr_prediction_mismatches"] for r in selected_rows)
        print(f"{method} vs BRR | max relative error={worst:.3g} | mismatches={mismatches}")
        if worst > args.parity_tol or mismatches:
            warnings.warn(f"{method}: inspect BRR parity before interpreting accuracy", RuntimeWarning)
    print(f"Saved results in: {out.resolve()}")
    print("sessions.csv | summary.csv | registration_plan.json | afe_projection.npy (if enabled)")



if __name__ == "__main__":
    main()
