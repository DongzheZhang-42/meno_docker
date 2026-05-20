# meno_docker — containerized NeMo + PEFT/LoRA on whispered ASR

Docker-first version of the FastConformer + PEFT pipeline.
Designed to run on **modern GPUs that the bare-metal lockfile cannot reach** —
including Blackwell (RTX 5090, GB202, `sm_120`) where stock `torch 2.2.1+cu121`
has no kernel image.

## Why this exists

The companion repo `meno_zdz` pins `nemo_toolkit==1.23.0` + `torch==2.2.1+cu121`
+ 177 other packages. That stack is verified on A40 (Ampere) but **does not
support Blackwell** because the torch 2.2.1 wheels weren't compiled for
`sm_120`. Trying to run it on a 5090 yields:

```
RuntimeError: no kernel image is available for execution on the device
```

The fix is to use NVIDIA's NeMo container, which bundles a torch build that
matches the GPU architecture. This repo provides two paths:

| Path | What | When |
|---|---|---|
| **`./run_stock.sh`** | Run scripts inside `nvcr.io/nvidia/nemo:25.04` as-is (NeMo 2.x) | Try first |
| **`./run_custom.sh`** | Build a custom image that downgrades NeMo to 1.23 inside the container | Fallback if 2.x has API drift |

Both inherit the stock container's torch + CUDA + cuDNN + Transformer Engine
(Blackwell-ready), so 5090 works either way.

---

## Prerequisites on the host

The host machine needs:

```bash
# 1. NVIDIA driver new enough for CUDA in the container (>= 555 for nemo:25.04)
nvidia-smi   # check "Driver Version" and "CUDA Version" columns

# 2. Docker (or podman with docker compat)
docker --version

# 3. NVIDIA Container Toolkit (lets docker see GPUs)
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
# If the above prints your GPU list, you're good. If not, install:
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

If `docker` isn't available, ask IT to install `docker-ce` + `nvidia-container-toolkit`.
On HPC clusters that disallow Docker, swap to `podman` or `apptainer/singularity`
— the commands below translate almost 1:1.

---

## Quickstart — stock container path

```bash
# 1. Clone this repo on the GPU host
git clone <THIS-REPO-URL>
cd <repo-name>

# 2. Pull the NeMo image (one-time, ~15 GB)
docker pull nvcr.io/nvidia/nemo:25.04

# 3. Enter the container, GPU 0
chmod +x run_stock.sh
./run_stock.sh 0
# (or ./run_stock.sh 2 to use GPU index 2)

# Now you're inside the container, in /workspace which is bind-mounted to
# this repo on the host. Anything you write here also lives on the host.

# 4. Inside the container — run the pipeline
python scripts/smoke_test.py                      # verify env
python scripts/download_model.py --out outputs/pretrained
python scripts/prepare_data.py --dataset librispeech --out manifests --max-per-split 200
python scripts/eval_wer.py \
    --model outputs/pretrained/model.nemo \
    --manifest manifests/test.json \
    --output outputs/baseline_hyps.json \
    --decoder rnnt --batch-size 8
python scripts/train_peft.py --config configs/peft_train.yaml
python scripts/eval_wer.py \
    --model outputs/pretrained/model.nemo \
    --adapter outputs/peft/checkpoints/adapter_best.nemo \
    --manifest manifests/test.json \
    --output outputs/peft_hyps.json
```

### If a script errors with `ImportError` or `AttributeError`

NeMo 2.x reorganized some module paths and renamed a few adapter helpers.
The scripts in this repo target the 1.23 API (where they were verified).
Most of the API surface is the same, but if you hit:

```python
ImportError: cannot import name 'ConformerEncoderAdapter' from ...
# or
AttributeError: 'EncDecHybridRNNTCTCBPEModel' object has no attribute 'add_adapter'
```

stop the stock container and switch to the custom one (below).

---

## Quickstart — custom NeMo 1.23 image

```bash
chmod +x run_custom.sh
./run_custom.sh 0

# First invocation triggers `docker build -f docker/Dockerfile .` which takes
# 10–20 minutes (compiles youtokentome from source, downgrades NeMo & deps).
# Subsequent invocations reuse the local image — instant.
```

The custom image preserves the stock container's torch (Blackwell-capable)
but force-installs `nemo_toolkit==1.23.0` and the exact 1.23-compatible
deps from `meno_zdz/env/requirements.lock.txt`. The scripts run unchanged.

---

## Repo layout

```
.
├── README.md
├── .gitignore
├── docker/
│   └── Dockerfile              # Custom NeMo 1.23 image recipe (run_custom)
├── run_stock.sh                # Enter stock nvcr.io/nvidia/nemo:25.04
├── run_custom.sh               # Build + enter custom 1.23 image
├── configs/
│   └── peft_train.yaml         # Training hyperparameters
└── scripts/
    ├── smoke_test.py
    ├── download_model.py
    ├── prepare_data.py
    ├── eval_wer.py
    ├── train_peft.py
    ├── inspect_model.py
    └── check_encoder.py
```

`outputs/`, `manifests/`, `.cache_hf/`, `.cache_nemo/` are all gitignored and
appear when scripts run.

---

## Troubleshooting

**`docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]`**
→ NVIDIA Container Toolkit not installed. See prerequisites.

**`Unable to find image 'nvcr.io/nvidia/nemo:25.04' locally`** + pull hangs
→ Corporate firewall blocking `nvcr.io`. Either ask IT for a mirror or
`docker save` the image on a machine that can reach NGC and `docker load`
on the company machine (`docker save nvcr.io/nvidia/nemo:25.04 | gzip > nemo.tar.gz`,
~10 GB compressed).

**`RuntimeError: no kernel image is available for execution`** inside the container
→ The container's torch can't target this GPU. Either the image tag is too
old (try `25.07` or later) or the GPU is too new even for the latest tag.

**Out of shared memory / NCCL errors** during data loading
→ Add or increase `--shm-size=16g` in `run_*.sh` (already set to 8g).

**Adapter `__class__` swap fails in stock container (NeMo 2.x)**
→ NeMo 2.x merged `ConformerEncoderAdapter` into `ConformerEncoder`. The
swap is now a no-op — `train_peft.py` already guards with an
`isinstance` check, so the line silently skips on 2.x. If you get a
different error, switch to `run_custom.sh`.

---

## Why containers solve what they solve (and what they don't)

✅ **Solved**: Python dep hell, CUDA/torch/cuDNN matching, GPU architecture
support, "works on my machine" reproducibility, NeMo's stricter version
pinning.

❌ **Not solved**: GPU driver on the host (must be new enough), network
access to pull the image, disk space (~15–25 GB per image), corporate
docker-disallowed policies, and—obviously—your code logic, hyperparameter
choices, and data quality.

The container is environment-as-data: it eliminates the part of "running
ML code on a new machine" that has nothing to do with the ML itself.
