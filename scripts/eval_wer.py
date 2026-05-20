"""Evaluate WER of a NeMo ASR model on a manifest.

Runs RNNT decoder by default (hybrid model also has a CTC head — pass
--decoder ctc to compare).

If --adapter is given, loads the adapter checkpoint on top of the base model
before decoding — that's how we measure the LoRA/PEFT improvement.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help=".nemo file (base model)")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="Optional adapter-only checkpoint to layer on top")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True,
                    help="Where to dump per-utt hypotheses + WER summary")
    ap.add_argument("--decoder", choices=["rnnt", "ctc"], default="rnnt")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from jiwer import wer

    print(f"[load] {args.model}")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(str(args.model), map_location="cpu")

    if args.adapter is not None:
        # Same encoder class swap as in train_peft.py — needed so the model
        # has the add/load adapter methods on its encoder.
        from nemo.collections.asr.modules import ConformerEncoder, ConformerEncoderAdapter
        if isinstance(model.encoder, ConformerEncoder) and not isinstance(model.encoder, ConformerEncoderAdapter):
            model.encoder.__class__ = ConformerEncoderAdapter
            print("[patch] encoder class → ConformerEncoderAdapter")
        print(f"[load] adapter {args.adapter}")
        model.load_adapters(str(args.adapter))
        model.set_enabled_adapters(enabled=True)

    model.change_decoding_strategy(decoder_type=args.decoder)
    model = model.to(args.device).eval()

    # Read manifest
    utts = []
    with args.manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                utts.append(json.loads(line))
    print(f"[eval] {len(utts)} utterances, decoder={args.decoder}")

    audio_paths = [u["audio_filepath"] for u in utts]
    refs = [u["text"] for u in utts]

    with torch.no_grad():
        hyps = model.transcribe(audio_paths, batch_size=args.batch_size)

    # NeMo's transcribe returns (best, all) for some versions; normalize.
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    # For hybrid model RNNT, items can be Hypothesis objects — extract .text
    hyps = [h.text if hasattr(h, "text") else str(h) for h in hyps]
    hyps_norm = [_norm(h) for h in hyps]
    refs_norm = [_norm(r) for r in refs]

    overall_wer = wer(refs_norm, hyps_norm)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({
            "model": str(args.model),
            "adapter": str(args.adapter) if args.adapter else None,
            "decoder": args.decoder,
            "manifest": str(args.manifest),
            "num_utts": len(utts),
            "wer": overall_wer,
            "per_utt": [
                {"audio": a, "ref": r, "hyp": h, "wer": wer([r], [h])}
                for a, r, h in zip(audio_paths, refs_norm, hyps_norm)
            ],
        }, f, indent=2, ensure_ascii=False)

    print(f"[wer] {overall_wer:.4f}  → {args.output}")


def _norm(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


if __name__ == "__main__":
    main()
