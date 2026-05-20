"""PEFT (LoRA-style adapter) fine-tuning for NeMo FastConformer hybrid.

Pipeline:

  1. Load pretrained .nemo
  2. Add an adapter to the encoder (NeMo's built-in LinearAdapter; the
     LoRA-specific subclass is used when available in this NeMo version)
  3. Freeze backbone, unfreeze only adapter params
  4. Configure train/val dataloaders from manifests
  5. Run Lightning trainer
  6. Save adapter-only checkpoint

For a pure HF-PEFT LoRA branch (manually wrapping ConformerLayer q/k/v/o), see
the comment block at the bottom of this file — kept as a sketch because the
NeMo Lightning loop holds the optimizer + saving paths, and rewiring those
isn't free.
"""
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf, DictConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "resolved_config.yaml")

    import torch
    import pytorch_lightning as pl
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from nemo.utils.exp_manager import exp_manager

    print(f"[load] {cfg.base_model}")
    model = EncDecHybridRNNTCTCBPEModel.restore_from(str(cfg.base_model), map_location="cpu")

    # The streaming FastConformer checkpoint ships with a bare ConformerEncoder
    # that doesn't inherit AdapterModuleMixin. ConformerEncoderAdapter is a
    # method-only subclass (no extra state) and ConformerLayer is already
    # adapter-compatible, so swapping __class__ is a safe in-place upgrade.
    from nemo.collections.asr.modules import ConformerEncoder, ConformerEncoderAdapter
    if isinstance(model.encoder, ConformerEncoder) and not isinstance(model.encoder, ConformerEncoderAdapter):
        model.encoder.__class__ = ConformerEncoderAdapter
        print("[patch] encoder class → ConformerEncoderAdapter (adapter-enabled)")

    # ----- Attach adapter -----
    adapter_cfg = _build_adapter_cfg(model, cfg.adapter)
    adapter_full_name = f"{cfg.adapter.module}:{cfg.adapter.name}"
    print(f"[adapter] adding {adapter_full_name}  type={cfg.adapter.type}  dim={cfg.adapter.dim}")
    model.add_adapter(name=adapter_full_name, cfg=adapter_cfg)
    model.set_enabled_adapters(enabled=True)

    # Freeze base, train only adapter params
    model.freeze()
    model.unfreeze_enabled_adapters()
    _report_trainable(model)

    # ----- Dataloaders from manifests -----
    train_ds = OmegaConf.create({
        "manifest_filepath": str(cfg.manifests.train),
        "sample_rate": 16000,
        "batch_size": cfg.train.batch_size,
        "shuffle": True,
        "num_workers": cfg.train.num_workers,
        "pin_memory": True,
        "use_start_end_token": False,
        "trim_silence": False,
        "max_duration": 20.0,
        "min_duration": 0.1,
    })
    val_ds = OmegaConf.create({
        "manifest_filepath": str(cfg.manifests.val),
        "sample_rate": 16000,
        "batch_size": cfg.train.batch_size,
        "shuffle": False,
        "num_workers": cfg.train.num_workers,
        "pin_memory": True,
        "use_start_end_token": False,
    })
    model.setup_training_data(train_data_config=train_ds)
    model.setup_validation_data(val_data_config=val_ds)

    # Compute max_steps now (used by both scheduler and Trainer below)
    n_train = sum(1 for _ in open(cfg.manifests.train, encoding="utf-8") if _.strip())
    steps_per_epoch = max(1, n_train // cfg.train.batch_size)
    max_steps = steps_per_epoch * cfg.train.max_epochs
    print(f"[sched] train_utts={n_train} steps/epoch={steps_per_epoch} max_steps={max_steps}")

    # ----- Optimizer (only trainable params get updates) -----
    sched_dict = dict(cfg.optim.sched)
    sched_dict["max_steps"] = max_steps
    model.setup_optimization(optim_config=OmegaConf.create({
        "name": cfg.optim.name,
        "lr": cfg.optim.lr,
        "weight_decay": cfg.optim.weight_decay,
        "sched": sched_dict,
    }))

    # ----- Trainer -----
    # logger=False because exp_manager will attach its own loggers below
    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,
        max_steps=max_steps,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.train.precision,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        log_every_n_steps=10,
        enable_checkpointing=False,  # exp_manager will install its own callback
        logger=False,
        default_root_dir=str(out_dir),
    )
    # NeMo's exp_manager handles per-epoch best-checkpoint tracking
    exp_manager(trainer, OmegaConf.create({
        "exp_dir": str(out_dir),
        "name": "peft",
        "create_checkpoint_callback": True,
        "checkpoint_callback_params": {
            "monitor": cfg.output.monitor,
            "mode": cfg.output.monitor_mode,
            "save_top_k": cfg.output.save_top_k,
            "always_save_nemo": False,
        },
    }))
    model.set_trainer(trainer)

    trainer.fit(model)

    # ----- Save adapter-only checkpoint -----
    adapter_path = out_dir / "checkpoints" / "adapter_best.nemo"
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    # NeMo's save_to writes full model+adapter; for adapter-only, use save_adapters
    if hasattr(model, "save_adapters"):
        model.save_adapters(str(adapter_path))
        print(f"[save] adapter-only weights → {adapter_path}")
    else:
        full_path = adapter_path.with_name("model_with_adapter.nemo")
        model.save_to(str(full_path))
        print(f"[save] (no save_adapters in this NeMo) full model → {full_path}")


def _build_adapter_cfg(model, ad_cfg: DictConfig):
    """Construct the adapter config matching ad_cfg.type. Falls back gracefully."""
    # Detect encoder d_model
    d_model = None
    for attr in ("d_model", "feat_in", "feat_out"):
        if hasattr(model.encoder, attr):
            v = getattr(model.encoder, attr)
            if isinstance(v, int):
                d_model = v
                break
    if d_model is None:
        # FastConformer-large default
        d_model = 512
    print(f"[adapter] encoder d_model = {d_model}")

    if ad_cfg.type == "lora":
        # NeMo 1.23+ ships a LoRA-style adapter for ASR linear modules.
        # Module path moved between versions; try both.
        for path in (
            "nemo.collections.common.parts.adapter_modules",
            "nemo.collections.asr.parts.submodules.adapters",
        ):
            try:
                mod = __import__(path, fromlist=["*"])
            except ImportError:
                continue
            for cls_name in ("LoraAdapterConfig", "LoRAAdapterConfig"):
                cls = getattr(mod, cls_name, None)
                if cls is not None:
                    return cls(in_features=d_model, dim=ad_cfg.dim,
                               dropout=ad_cfg.dropout)
        print("[adapter] LoRA adapter cfg not found in this NeMo build — "
              "falling back to LinearAdapterConfig (Houlsby).")

    from nemo.collections.common.parts import adapter_modules
    return adapter_modules.LinearAdapterConfig(
        in_features=d_model,
        dim=ad_cfg.dim,
        dropout=ad_cfg.dropout,
    )


def _report_trainable(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / max(total, 1)
    print(f"[params] trainable {trainable:,} / {total:,}  ({pct:.3f}%)")


# -------------------------------------------------------------------------
# Sketch: pure HF-PEFT LoRA branch (NOT wired into main flow)
# -------------------------------------------------------------------------
# from peft import LoraConfig, get_peft_model
# # Inspect: grep model.named_modules() for ConformerLayer's q/k/v/o Linears.
# # On NeMo FastConformer they live under e.g.
# #   encoder.layers.{i}.self_attn.linear_q / linear_k / linear_v / linear_out
# lora_cfg = LoraConfig(
#     r=16, lora_alpha=32, lora_dropout=0.1,
#     target_modules=["linear_q", "linear_k", "linear_v", "linear_out"],
#     bias="none", task_type=None,
# )
# model_with_lora = get_peft_model(model, lora_cfg)
# # Then you must (a) ensure NeMo's setup_optimization picks up only
# # trainable params, and (b) override save/load to handle the PEFT state dict.
# # This is workable but takes ~half a day of plumbing — only do it if the
# # team specifically rejects NeMo's native adapter.


if __name__ == "__main__":
    main()
