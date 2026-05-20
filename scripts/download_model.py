"""Download the NeMo pretrained FastConformer hybrid streaming model.

NeMo's from_pretrained API hits NGC by default; we save a local copy so subsequent
scripts don't need network access.
"""
import argparse
from pathlib import Path

MODEL_NAME = "stt_en_fastconformer_hybrid_large_streaming_80ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/pretrained"))
    ap.add_argument("--model", default=MODEL_NAME,
                    help="NGC model name (default: streaming FastConformer hybrid large 80ms)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / "model.nemo"

    if dst.exists():
        print(f"[skip] {dst} already exists ({dst.stat().st_size / 1e6:.1f} MB)")
        return

    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    print(f"[download] from_pretrained('{args.model}') …")
    model = EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=args.model)
    model.save_to(str(dst))
    print(f"[done] saved → {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
