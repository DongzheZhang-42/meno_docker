"""Print the FastConformer encoder structure — useful for picking LoRA target_modules.

Run after download_model.py to see exact module names you'd hand to
peft.LoraConfig(target_modules=[...]).
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("outputs/pretrained/model.nemo"))
    ap.add_argument("--filter", default="linear",
                    help="Only print modules whose name contains this substring")
    args = ap.parse_args()

    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    model = EncDecHybridRNNTCTCBPEModel.restore_from(str(args.model), map_location="cpu")

    print(f"[model] encoder class: {type(model.encoder).__name__}")
    for attr in ("d_model", "feat_in"):
        if hasattr(model.encoder, attr):
            print(f"        {attr} = {getattr(model.encoder, attr)}")

    print(f"\n[modules] containing '{args.filter}':\n")
    seen_kinds = set()
    for name, module in model.named_modules():
        if args.filter.lower() not in name.lower():
            continue
        kind = type(module).__name__
        # Only print one example of each unique (kind, depth-pattern) combo
        depth = name.count(".")
        key = (kind, depth, ".".join(name.split(".")[:-1])[:60])
        if key in seen_kinds:
            continue
        seen_kinds.add(key)
        print(f"  {name:60s}  ({kind})")

    print("\n[hint] For HF-PEFT LoRA, hand the leaf-module names (last path segment) "
          "as target_modules, e.g. ['linear_q','linear_k','linear_v','linear_out'].")


if __name__ == "__main__":
    main()
