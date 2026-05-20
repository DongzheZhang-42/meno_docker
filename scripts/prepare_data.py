"""Build NeMo manifests for whispered-speech ASR fine-tuning.

Two dataset paths:

  --dataset chains       Real whispered speech (CHAINS WHSP subset).
                         User must FTP-download from chains.ucd.ie/ftpaccess.php
                         and unpack to --root. We walk the wav files, match each
                         to its prompt text via the corpus's CHAINS_PROMPTS map,
                         and emit train/val/test manifests split BY SPEAKER so
                         the test set has no speaker overlap with train.

  --dataset librispeech  LibriSpeech dev-clean + a pseudo-whisper augmentation
                         (LPC residual / source-filter de-voicing). Fully
                         automatic via HF `datasets`. Use this for smoke-testing
                         the PEFT pipeline before downloading CHAINS.

Output: manifests/{train,val,test}.json — newline-delimited JSON, NeMo format:
    {"audio_filepath": "...", "duration": float, "text": "..."}
"""
import argparse
import json
import re
from pathlib import Path

# CHAINS prompts: each whispered utterance file is named like <SPK>_<UTT>.wav.
# The UTT id maps to a fixed prompt from one of the fables / CSLU sentences.
# Source: CHAINS overview doc (chains.ucd.ie/docs/chains_corpus_specom2006.pdf).
# We embed only the structural mapping; the actual prompt strings come from a
# transcripts file that ships inside the CHAINS distribution under doc/.
# If --transcripts is omitted we read doc/transcripts.txt from --root.
CHAINS_TRANSCRIPT_RELPATH = "doc/transcripts.txt"


def load_chains_prompts(root: Path, transcripts_path: Path | None):
    """Return dict: utt_id -> normalized text. Tolerant to two doc formats."""
    path = transcripts_path or (root / CHAINS_TRANSCRIPT_RELPATH)
    if not path.exists():
        raise FileNotFoundError(
            f"CHAINS transcripts file not found at {path}. "
            "Pass --transcripts to point at the corpus's prompts file."
        )

    prompts = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Tolerate either '<id>\t<text>' or '<id>: <text>' formats.
            m = re.match(r"^([A-Za-z0-9_]+)\s*[:\t]\s*(.+)$", line)
            if m:
                utt_id, text = m.group(1), m.group(2)
                prompts[utt_id.lower()] = normalize_text(text)
    if not prompts:
        raise ValueError(f"No prompts parsed from {path} — check format.")
    return prompts


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation NeMo's BPE tokenizer doesn't like, collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def wav_duration(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


def build_chains_manifests(root: Path, out: Path, transcripts: Path | None,
                           split_by_speaker: bool):
    prompts = load_chains_prompts(root, transcripts)
    whsp_root = root / "WHSP"
    if not whsp_root.exists():
        # Some unpacked layouts put it at root/data/WHSP
        alt = root / "data" / "WHSP"
        if alt.exists():
            whsp_root = alt
        else:
            raise FileNotFoundError(f"No WHSP dir under {root}")

    rows = []
    unmatched = 0
    for wav in sorted(whsp_root.rglob("*.wav")):
        # Filename convention: <SPK>_<UTT>.wav  (e.g. s01_f01.wav)
        stem = wav.stem.lower()
        m = re.match(r"^([a-z]\d{2})_(.+)$", stem)
        if not m:
            unmatched += 1
            continue
        speaker, utt = m.group(1), m.group(2)
        text = prompts.get(utt)
        if text is None:
            unmatched += 1
            continue
        rows.append({
            "audio_filepath": str(wav.resolve()),
            "duration": round(wav_duration(wav), 3),
            "text": text,
            "speaker": speaker,
        })

    print(f"[chains] matched {len(rows)} utterances, {unmatched} unmatched")
    if not rows:
        raise RuntimeError("No utterances matched — check filename convention & transcripts.")

    if split_by_speaker:
        speakers = sorted({r["speaker"] for r in rows})
        # 70/15/15 by speaker, deterministic
        n = len(speakers)
        train_spk = set(speakers[: int(n * 0.7)])
        val_spk = set(speakers[int(n * 0.7) : int(n * 0.85)])
        test_spk = set(speakers[int(n * 0.85) :])
    else:
        # Random utterance-level split (NOT recommended — leaks speakers)
        import random
        random.seed(0)
        random.shuffle(rows)
        n = len(rows)
        train_idx = set(range(int(n * 0.7)))
        val_idx = set(range(int(n * 0.7), int(n * 0.85)))
        # test = rest

    splits = {"train": [], "val": [], "test": []}
    for i, r in enumerate(rows):
        if split_by_speaker:
            spk = r["speaker"]
            if spk in train_spk:
                splits["train"].append(r)
            elif spk in val_spk:
                splits["val"].append(r)
            elif spk in test_spk:
                splits["test"].append(r)
        else:
            if i in train_idx:
                splits["train"].append(r)
            elif i in val_idx:
                splits["val"].append(r)
            else:
                splits["test"].append(r)

    write_manifests(splits, out)


def build_librispeech_manifests(out: Path, pseudo_whisper: bool, max_per_split: int):
    """Fallback: LibriSpeech dev-clean via HF datasets, optionally de-voiced."""
    from datasets import load_dataset
    import soundfile as sf
    import numpy as np

    cache_dir = out.parent / "cache_librispeech"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print("[librispeech] loading dev-clean from HF …")
    ds = load_dataset("openslr/librispeech_asr", "clean", split="validation",
                      cache_dir=str(cache_dir))

    # Speaker-disjoint split
    speakers = sorted({ex["speaker_id"] for ex in ds})
    n = len(speakers)
    train_spk = set(speakers[: int(n * 0.7)])
    val_spk = set(speakers[int(n * 0.7) : int(n * 0.85)])

    splits = {"train": [], "val": [], "test": []}
    wav_dir = out.parent / "librispeech_wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    for ex in ds:
        audio = ex["audio"]["array"].astype("float32")
        sr = ex["audio"]["sampling_rate"]
        if pseudo_whisper:
            audio = _pseudo_whisper(audio, sr)

        wav_path = wav_dir / f"{ex['id']}.wav"
        sf.write(str(wav_path), audio, sr, subtype="PCM_16")

        row = {
            "audio_filepath": str(wav_path.resolve()),
            "duration": round(len(audio) / sr, 3),
            "text": normalize_text(ex["text"]),
        }
        spk = ex["speaker_id"]
        bucket = "train" if spk in train_spk else "val" if spk in val_spk else "test"
        if len(splits[bucket]) < max_per_split:
            splits[bucket].append(row)
        if all(len(v) >= max_per_split for v in splits.values()):
            break

    write_manifests(splits, out)


def _pseudo_whisper(audio, sr):
    """Crude whisperization: high-pass + LPC-residual emphasis.

    Real whispered speech has no voiced excitation (no F0). We approximate this
    by extracting the LPC residual (which removes the formant envelope), then
    re-adding a fraction of the envelope. This is NOT a substitute for real
    whispered data — it's just enough perturbation that the LoRA branch sees
    a domain shift it has to adapt to. Smoke-test only.
    """
    import numpy as np
    from scipy.signal import lfilter, butter

    # High-pass to attenuate F0
    b, a = butter(4, 200 / (sr / 2), btype="high")
    audio = lfilter(b, a, audio).astype("float32")

    # Inverse filtering via 12-order LPC, keep residual
    order = 12
    try:
        import librosa
        lpc = librosa.lpc(audio, order=order)
        residual = lfilter(lpc, [1.0], audio)
        # Blend residual + low-amplitude original for breathy quality
        audio = 0.7 * residual + 0.3 * audio
    except Exception:
        pass  # fall back to high-passed signal

    # Re-normalize
    peak = np.max(np.abs(audio)) + 1e-9
    return (audio / peak * 0.95).astype("float32")


def write_manifests(splits: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        p = out / f"{name}.json"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dur = sum(r["duration"] for r in rows)
        print(f"[write] {p}  {len(rows)} utts, {dur/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["chains", "librispeech"], required=True)
    ap.add_argument("--root", type=Path, help="CHAINS unpacked root (required for chains)")
    ap.add_argument("--transcripts", type=Path,
                    help="Path to CHAINS prompts file (default: <root>/doc/transcripts.txt)")
    ap.add_argument("--out", type=Path, default=Path("manifests"))
    ap.add_argument("--split-by-speaker", action="store_true",
                    help="(chains) split by speaker, not utterance — recommended")
    ap.add_argument("--no-pseudo-whisper", action="store_true",
                    help="(librispeech) skip de-voicing augmentation")
    ap.add_argument("--max-per-split", type=int, default=200,
                    help="(librispeech) cap utterances per split for fast smoke test")
    args = ap.parse_args()

    if args.dataset == "chains":
        if not args.root:
            ap.error("--root is required for --dataset chains")
        build_chains_manifests(args.root, args.out, args.transcripts, args.split_by_speaker)
    else:
        build_librispeech_manifests(args.out, not args.no_pseudo_whisper, args.max_per_split)


if __name__ == "__main__":
    main()
