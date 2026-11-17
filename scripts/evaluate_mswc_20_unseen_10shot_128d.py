"""Evaluate frozen 128D CNN on 20 genuinely unseen English MSWC words.
使用20个完全未见过的MSWC英语开发词，评价冻结的128维CNN的10-shot识别能力。

Run from E:\\IP_Incremental_Learning / 从项目根目录运行。
No CNN training, official en_test.csv, or final-heldout classes are used.
不训练CNN，不读取官方en_test.csv，也不读取最终保留测试词。

The fixed support pool has 40 recordings/word from official TRAIN; the fixed
query set has 50 recordings/word from official DEV. Each seed picks 10 supports
from those 40, makes one Euclidean prototype per word, and classifies all
20*50=1000 development query examples in a 20-way task.
注册池为官方TRAIN每词40条；查询集为官方DEV每词50条。每个随机种子从
40条中抽10条计算每词Euclidean原型，并评价20类×50条=1000条查询录音。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import nn

# Fixed, frozen data selection / 冻结的数据划分；不可用最终测试集调参。
ROOT = Path('data/prepared/mswc_large_vocab_128d')
SELECT = ROOT / 'sample_selection'
EXTRACTED = Path('data/external/mswc_full_en/extracted')
VOCAB = ROOT / 'vocabulary_manifest.json'
CACHE = ROOT / 'unseen_dev_audio_cache_1p5s'
RESULTS = Path('results/mswc_unseen_20word_10shot')
MODEL_LARGE = Path('models/feature_extractor_128d_mswc_1005class_128d_1p5s_ce_formal.pth')
MODEL_OLD = Path('models/feature_extractor_128d_no_relu_aug_31class_fast.pth')
SAMPLE_RATE = 16000
SEEDS = (42, 123, 2026, 7, 99)
SHOT = 10
BATCH = 64


def read_manifest(filename: str, expected_each: int, names: list[str], required_split: str):
    """Read immutable selection, verify IDs, labels, splits, counts and uniqueness.
    读取原始固定清单，校验单词、官方划分、数量、路径与重复录音。
    """
    source = SELECT / filename
    if not source.is_file():
        raise FileNotFoundError(f'Missing fixed selection CSV: {source}')
    groups = defaultdict(list)
    seen = set()
    with source.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'word', 'tar_member', 'speaker', 'official_split'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'{source}: missing CSV fields {required - set(reader.fieldnames or ())}')
        for line, row in enumerate(reader, 2):
            word = row['word'].strip().lower()
            raw = row['tar_member'].replace('\\', '/').strip()
            p = PurePosixPath(raw)
            if word not in names or row['official_split'].strip().lower() != required_split:
                raise ValueError(f'{source}:{line}: unexpected word or split')
            if (p.is_absolute() or '..' in p.parts or len(p.parts) != 4 or
                    p.parts[:3] != ('en', 'clips', word) or p.suffix.lower() != '.opus'):
                raise ValueError(f'{source}:{line}: unsafe/invalid Opus path: {raw}')
            if raw in seen:
                raise ValueError(f'{source}:{line}: duplicated recording {raw}')
            seen.add(raw)
            groups[word].append((raw, row['speaker'].strip()))
    for word in names:
        if len(groups[word]) != expected_each:
            raise ValueError(f'{source}: {word} has {len(groups[word])}, expected {expected_each}')
    return groups


def decode_or_load(rel: str, samples: int) -> np.ndarray:
    """Cache only 1800 DEV clips, NOT 1.9 million training recordings.
    只缓存1800条开发音频，绝不再次处理191万条大规模预训练语料。

    FFmpeg is used once for each cache miss; subsequent evaluations read WAV.
    每条缺失音频首次用FFmpeg解码，重复评价直接读取缓存WAV。
    """
    relative = PurePosixPath(rel)
    opus = EXTRACTED.joinpath(*relative.parts)
    wav = CACHE.joinpath(*relative.with_suffix('.wav').parts)
    if wav.is_file():
        audio, rate = sf.read(str(wav), dtype='float32', always_2d=False)
        if rate == SAMPLE_RATE and audio.ndim == 1 and audio.shape[0] == 24000:
            return audio[:samples].copy()
        raise ValueError(f'Existing DEV cache has invalid format: {wav}')
    if not opus.is_file():
        raise FileNotFoundError(f'Expected original DEV Opus: {opus}')
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-i', str(opus), '-ac', '1', '-ar', str(SAMPLE_RATE),
               '-f', 'f32le', '-acodec', 'pcm_f32le', '-']
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=35, check=False)
    if proc.returncode:
        raise RuntimeError(f'FFmpeg failed for {opus}: {proc.stderr.decode(errors="replace")[:400]}')
    audio = np.frombuffer(proc.stdout, dtype='<f4').copy()
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError(f'Empty/invalid audio: {opus}')
    # Keep original onset and pad at END; length > 1.5s is truncated.
    # 保留原始开头，结尾补零；超过1.5秒仍需裁剪，不能凭空恢复原录音外的发音。
    audio = np.pad(audio[:24000], (0, max(0, 24000 - min(len(audio), 24000))))
    wav.parent.mkdir(parents=True, exist_ok=True)
    temp = wav.with_suffix('.partial.wav')
    sf.write(str(temp), audio, SAMPLE_RATE, subtype='PCM_16')
    temp.replace(wav)
    return audio[:samples]


def feature_network() -> nn.Sequential:
    """EXACT same 128D SmallCNN feature architecture as old/new model.
    与新旧模型严格一致的128维特征提取器；末尾不添加ReLU。
    """
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 5)), nn.Flatten(), nn.Linear(320, 128)
    )


def extract_all(data: dict, names: list[str], model_path: Path, samples: int,
                device: torch.device, batch_size: int):
    """Freeze extractor and compute all features once; never update weights.
    冻结CNN权重，全部样本只提取一次特征，不反向传播也不更新网络。
    """
    if not model_path.is_file():
        raise FileNotFoundError(f'Best feature checkpoint missing: {model_path}')
    net = feature_network().to(device)
    state = torch.load(str(model_path), map_location='cpu', weights_only=True)
    # Our training convention stores only model.features.state_dict().
    # 训练脚本只保存features层权重，不能误传完整训练状态Checkpoint。
    net.load_state_dict(state, strict=True)
    net.eval()
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
              n_fft=400, hop_length=160, n_mels=40).to(device)
    result = {}
    with torch.inference_mode():
        for index, word in enumerate(names):
            paths = [record[0] for record in data[word]]
            chunks = []
            for begin in range(0, len(paths), batch_size):
                waves = np.stack([decode_or_load(p, samples)
                                  for p in paths[begin:begin+batch_size]])
                t = torch.from_numpy(waves).to(device, non_blocking=True)
                spec = torch.log(mel(t.unsqueeze(1)) + 1e-6) / 10.0
                features = net(spec).cpu().numpy().astype(np.float64)
                chunks.append(features)
            result[word] = np.concatenate(chunks, axis=0)
            if (index + 1) % 5 == 0:
                print(f'  Extracted features: {index+1}/{len(names)} words', flush=True)
    return result


def classify(queries: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Squared Euclidean nearest prototype; same baseline as existing work.
    使用平方欧氏距离和每类均值原型，不增加新训练参数。
    """
    d = ((queries ** 2).sum(axis=1, keepdims=True) +
         (prototypes ** 2).sum(axis=1)[None, :] - 2.0 * queries @ prototypes.T)
    return d.argmin(axis=1)


def save_csv(path: Path, rows: list[dict]):
    """Export result tables with Windows-compatible UTF-8 BOM.
    结果CSV带BOM，适合Windows Excel及VS Code读取。
    """
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description='20 unseen MSWC English words, 10-shot, 128D')
    parser.add_argument('--model', type=Path, default=MODEL_LARGE,
                        help='feature-only .pth, not last_training_state .pth')
    parser.add_argument('--compare-old', action='store_true',
                        help='also evaluate old 31-way 1s model using same recordings')
    parser.add_argument('--batch-size', type=int, default=BATCH)
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    args = parser.parse_args()
    if args.batch_size < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('invalid batch-size or repeated/empty seed list')
    info = json.loads(VOCAB.read_text(encoding='utf-8-sig'))
    words = [str(w).lower() for w in info['development_words']]
    pretrain = set(map(str.lower, info['train_words']))
    final = set(map(str.lower, info['final_heldout_words']))
    if len(words) != 20 or len(set(words)) != 20 or set(words) & (pretrain | final):
        raise RuntimeError('Vocabulary separation is not valid')
    support = read_manifest('novel_dev_support_pool.csv', 40, words, 'train')
    query = read_manifest('novel_dev_query.csv', 50, words, 'dev')
    support_paths = {p for entries in support.values() for p, _ in entries}
    query_paths = {p for entries in query.values() for p, _ in entries}
    support_speakers = {sp for entries in support.values() for _, sp in entries if sp}
    query_speakers = {sp for entries in query.values() for _, sp in entries if sp}
    if support_paths & query_paths or support_speakers & query_speakers:
        raise RuntimeError('Support/query recording or speaker overlap')
    # Also ensure NEW words cannot be present in 1005-way pretrain class list.
    # 再检查1005类清单，确认新词没有偷偷出现在CNN预训练阶段。
    class_csv = ROOT / 'full_vocab_1p5s/class_map.csv'
    if not class_csv.is_file():
        raise FileNotFoundError(class_csv)
    with class_csv.open('r', encoding='utf-8-sig', newline='') as f:
        trained_words = {row['word'].strip().lower() for row in csv.DictReader(f)}
    if set(words) & trained_words:
        raise RuntimeError('LEAKAGE: some unseen development words are in pretraining class map')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Protocol PASS | unseen words=20 | support=800 | queries=1000 | speaker overlap=0', flush=True)
    print(f'Device={device} | official TEST and final-heldout: NOT ACCESSED', flush=True)
    models = [('1005class_1p5s', args.model, 24000)]
    if args.compare_old:
        models.append(('31class_1s', MODEL_OLD, 16000))
    RESULTS.mkdir(parents=True, exist_ok=True)
    seed_rows, word_rows, all_summary = [], [], []
    for model_label, checkpoint, samples in models:
        print(f'\nMODEL: {model_label} | checkpoint={checkpoint}', flush=True)
        sfeat = extract_all(support, words, checkpoint, samples, device, args.batch_size)
        qfeat = extract_all(query, words, checkpoint, samples, device, args.batch_size)
        query_x = np.concatenate([qfeat[w] for w in words], axis=0)
        query_y = np.repeat(np.arange(len(words)), 50)
        per_seed = []
        for seed in args.seeds:
            rng = random.Random(seed)
            # Independent random enrollment for each word; deterministic seed.
            # 每词固定随机种子独立抽10条，保证不同模型用同一批注册样本。
            centers = np.stack([sfeat[w][rng.sample(range(40), SHOT)].mean(axis=0)
                                for w in words])
            predictions = classify(query_x, centers)
            overall = float(np.mean(predictions == query_y) * 100)
            per_word_acc = []
            for i, word in enumerate(words):
                correct = int(np.sum(predictions[i*50:(i+1)*50] == i))
                acc = 100 * correct / 50
                per_word_acc.append(acc)
                word_rows.append(dict(model=model_label, seed=seed, word=word,
                                      correct=correct, total=50, accuracy_pct=round(acc, 3)))
            macro = float(np.mean(per_word_acc))
            seed_rows.append(dict(model=model_label, seed=seed,
                                  accuracy_pct=round(overall, 3),
                                  macro_accuracy_pct=round(macro, 3)))
            per_seed.append(overall)
            print(f'  Seed={seed} | 20-way 10-shot: {overall:.2f}% | macro={macro:.2f}%', flush=True)
        avg, std = float(np.mean(per_seed)), float(np.std(per_seed))
        all_summary.append(dict(model=model_label, mean_accuracy_pct=round(avg, 3),
                                std_accuracy_pct=round(std, 3),
                                seeds=len(args.seeds), classes=20, shots=SHOT,
                                query_per_class=50, audio_samples=samples))
        print(f'FINAL {model_label}: {avg:.2f}% ± {std:.2f}% (std across seeds)', flush=True)
    save_csv(RESULTS / 'per_seed.csv', seed_rows)
    save_csv(RESULTS / 'per_word.csv', word_rows)
    save_csv(RESULTS / 'summary.csv', all_summary)
    print(f'RESULTS SAVED: {RESULTS} | final held-out NOT USED', flush=True)


if __name__ == '__main__':
    main()
