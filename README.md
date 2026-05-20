# meno_docker — Containerized NeMo + PEFT/LoRA on whispered ASR

Self-contained playbook. Run [`preflight.sh`](preflight.sh) first, match the
output to the **decision tree**, then follow the matching path. Designed for
modern GPUs (including Blackwell / RTX 5090, `sm_120`) where stock
`torch 2.2.1+cu121` has no kernel image.

---

## Step 0 — Pre-flight (do this first, ~2 minutes)

```bash
# Clone, enter, run the diagnostic
git clone https://github.com/DongzheZhang-42/meno_docker.git
cd meno_docker
bash preflight.sh
```

You'll see 6 sections of output, each with `✓` (good), `!` (warning), or
`✗` (broken). Save the output — match it to the decision tree below.

### Expected ideal output (everything green)

```
[1/6] GPU & driver
0, NVIDIA GeForce RTX 5090, 575.51.02, 32760 MiB, 32500 MiB
✓  driver 575.x >= 555 — Blackwell (RTX 5090) supported

[2/6] Docker daemon
✓  Docker version 27.x
✓  docker daemon reachable

[3/6] GPU passthrough into container
✓  --gpus flag works, container sees:
      NVIDIA GeForce RTX 5090

[4/6] Network egress
✓  200  https://github.com
✓  200  https://nvcr.io
✓  200  https://huggingface.co
...

[5/6] Disk space — need ~25 GB docker root
[6/6] User permissions — docker group ✓
```

---

## 决策树 — match preflight output to a path

| Section that failed | Path to follow | Cost |
|---|---|---|
| 全部 ✓ | **锦囊 A** | ~40 min |
| [2/6] docker not installed | **锦囊 B** — ask IT, then retry | days (IT lead time) |
| [3/6] container can't see GPU | **锦囊 C** — install Container Toolkit | hours |
| [1/6] driver < 555 (and GPU is 5090) | **锦囊 D** — driver upgrade | hours, needs sudo |
| [4/6] `nvcr.io` blocked | **锦囊 E** — offline image transfer | +30 min |
| [4/6] `huggingface.co` blocked but `nvcr.io` OK | **锦囊 F** — switch to manual data path | +20 min |
| Multiple failures | Open multiple 锦囊 in sequence | — |

Open the matching 锦囊 below.

---

## 锦囊 A — Happy path (everything works)

This is the 95% case if all preflight checks are green.

```bash
# Already cd'd into meno_docker/

# 1. Pull NeMo container (~15 GB, one-time, 5-30 min depending on network)
docker pull nvcr.io/nvidia/nemo:25.04

# 2. Enter container, GPU 0 (change to your free GPU index)
chmod +x run_stock.sh
./run_stock.sh 0
```

Now you're inside the container. Continue with **"Pipeline" section below**.

### Inside the container — 7 commands for the full pipeline

```bash
# 1) verify env
python scripts/smoke_test.py
# expect: torch + cuda available, nemo X.Y loaded, LinearAdapterConfig present

# 2) download pretrained model (~460 MB to outputs/pretrained/)
python scripts/download_model.py --out outputs/pretrained

# 3) build manifests (auto-downloads LibriSpeech dev-clean ~350 MB to .cache_hf/)
python scripts/prepare_data.py --dataset librispeech --out manifests --max-per-split 200

# 4) baseline WER (no adapter)
python scripts/eval_wer.py \
    --model outputs/pretrained/model.nemo \
    --manifest manifests/test.json \
    --output outputs/baseline_hyps.json \
    --decoder rnnt --batch-size 8
# expect: "[wer] 0.0xxx" — note this number

# 5) PEFT training (~7-15 min on a single modern GPU)
python scripts/train_peft.py --config configs/peft_train.yaml
# expect: outputs/peft/checkpoints/adapter_best.nemo (~5 MB)

# 6) fine-tuned WER (same eval, with adapter loaded)
python scripts/eval_wer.py \
    --model outputs/pretrained/model.nemo \
    --adapter outputs/peft/checkpoints/adapter_best.nemo \
    --manifest manifests/test.json \
    --output outputs/peft_hyps.json \
    --decoder rnnt --batch-size 8
# expect: "[wer] 0.0xxx" — compare with baseline

# 7) exit container — workspace files are still on host
exit
```

### If step 1 or 5 errors with `ImportError` / `AttributeError`

The stock container has NeMo 2.x; some adapter APIs were renamed.
Switch to the custom image (NeMo 1.23 forced):

```bash
# Outside container
chmod +x run_custom.sh
./run_custom.sh 0
# First run: docker build takes 10-20 min. Subsequent: instant.
# Then re-run steps 1-7 above.
```

---

## 锦囊 B — Docker not installed

You can't fix this yourself on a corporate machine. Email IT with this text:

> Please install Docker CE and NVIDIA Container Toolkit on this host.
> Reference docs:
> - https://docs.docker.com/engine/install/
> - https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
>
> Alternatively, `podman` + `nvidia-container-toolkit` works the same;
> or `apptainer/singularity` if this is an HPC environment.

While waiting, you can still test the bare-metal path on a different
machine (see the companion repo `meno_zdz`) — but it won't work on
Blackwell GPUs.

---

## 锦囊 C — Container can't see GPU (Toolkit missing)

Docker is installed but `--gpus all` test failed. Ask IT to install the
NVIDIA Container Toolkit:

```bash
# What IT runs (Debian/Ubuntu):
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then re-run `bash preflight.sh` to confirm.

---

## 锦囊 D — Driver too old for Blackwell

`nvidia-smi` shows driver < 555 and you have an RTX 5090. The driver needs
upgrade — this requires root on the host.

```bash
# What IT runs (assumes Ubuntu, kernel headers present):
sudo apt update
sudo apt install -y nvidia-driver-575     # or whatever latest in the repo
sudo reboot

# After reboot:
nvidia-smi   # confirm "Driver Version: 575.xx"
```

If apt doesn't have 575 yet, NVIDIA's official `.run` installer or the
graphics-drivers PPA has it. IT will know.

---

## 锦囊 E — `nvcr.io` blocked by corporate firewall

Pull the image on a machine that CAN reach NGC, then transfer offline.

**On a machine with NGC access** (e.g. a personal laptop, or another lab
machine):

```bash
docker pull nvcr.io/nvidia/nemo:25.04
docker save nvcr.io/nvidia/nemo:25.04 | gzip > nemo-25.04.tar.gz
# Result: ~10 GB file

# Upload to GitHub release (public repo, large file OK up to 2 GB per file;
# for ~10 GB, split):
split -b 1900M nemo-25.04.tar.gz nemo-25.04.tar.gz.part-
# Create release on github.com/<your>/meno_docker → Releases → New release
# Attach all part files
```

**On the corporate machine** (which can reach github.com but not nvcr.io):

```bash
# Download all parts from the GitHub release page
cat nemo-25.04.tar.gz.part-* > nemo-25.04.tar.gz
gunzip -c nemo-25.04.tar.gz | docker load
# Image now appears in `docker images`, no nvcr.io needed
```

Then continue with 锦囊 A from "enter container" step.

---

## 锦囊 F — `huggingface.co` blocked but NGC OK

`prepare_data.py --dataset librispeech` needs HF to download LibriSpeech.
Two workarounds:

**Option 1**: Use the manual whispered-speech dataset path (no HF needed).
Download the CHAINS corpus from http://chains.ucd.ie/ftpaccess.php on
any machine with internet, unpack to `data/CHAINS/`, then:

```bash
# Inside container
python scripts/prepare_data.py --dataset chains --root data/CHAINS \
    --out manifests --split-by-speaker
```

**Option 2**: Pre-build manifests + audio on a different machine, transfer
the `manifests/` directory via GitHub release. The `audio_filepath` fields
in the JSON files are absolute paths — if the destination directory differs,
update the paths with `sed -i 's|/old/path|/new/path|g' manifests/*.json`.

---

## Pipeline reference (what the scripts do)

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `smoke_test.py` | — | env sanity check (stdout) |
| 2 | `download_model.py --out X` | NGC | `X/model.nemo` (~460 MB) |
| 3 | `prepare_data.py --dataset librispeech` | HF | `manifests/{train,val,test}.json` |
| 3' | `prepare_data.py --dataset chains --root X` | unpacked CHAINS | same |
| 4 | `eval_wer.py --model M --manifest T` | M + T | `--output` JSON + WER on stdout |
| 5 | `train_peft.py --config C` | `outputs/pretrained/model.nemo` + manifests | `outputs/peft/checkpoints/adapter_best.nemo` |
| 6 | `eval_wer.py --model M --adapter A --manifest T` | M + A + T | `--output` JSON + WER |

All paths in commands are relative to the container's `/workspace`, which
is bind-mounted from your host's `meno_docker/` directory. Edits and outputs
are visible on both sides.

---

## Troubleshooting (errors → fixes)

| Error message | Likely cause | Fix |
|---|---|---|
| `RuntimeError: no kernel image is available for execution` | torch in container doesn't have your GPU's compute capability | Use newer NeMo image tag (`25.07`, `25.09`); driver also must be ≥ 555 for 5090 |
| `could not select device driver "" with capabilities: [[gpu]]` | NVIDIA Container Toolkit missing | 锦囊 C |
| `permission denied while trying to connect to the Docker daemon` | User not in docker group | `sudo docker ...` for now; ask IT to add you to `docker` group |
| `Pulling fs layer` hangs forever | nvcr.io blocked | 锦囊 E |
| `ImportError: cannot import name 'ConformerEncoderAdapter'` inside container | NeMo 2.x API drift | Use `run_custom.sh` instead of `run_stock.sh` |
| `OSError: [Errno 28] No space left on device` | docker root full | `docker system prune -a` or move docker root to bigger disk |
| OOM (CUDA out of memory) during training | Batch too big for GPU | Edit `configs/peft_train.yaml`: `batch_size: 4`, `accumulate_grad_batches: 2` |
| `KeyError` in `setup_training_data` | Manifest field mismatch | Check manifest JSON: each line needs `audio_filepath`, `duration`, `text` |
| `Trainer.fit stopped: max_steps=0 reached` | Empty train manifest | Verify `manifests/train.json` has > 0 lines |

---

## What the playbook does NOT cover (be aware)

- **Hyperparameter tuning** — `configs/peft_train.yaml` has defaults that
  work on the demo. Real fine-tuning may need to sweep `adapter.dim`,
  `optim.lr`, `train.max_epochs`.
- **Multi-GPU training** — current `train_peft.py` uses 1 GPU. To use
  DDP, change `devices=1` in `train_peft.py` to `devices=N` and add the
  `strategy="ddp"` arg.
- **Your own data** — if you skip `prepare_data.py` and bring your own
  NeMo manifests, point `configs/peft_train.yaml` at them.
- **WER interpretation** — a low baseline (~2%) on a benign test set
  doesn't mean fine-tuning is useless; on a real domain-shifted test set,
  expect baseline much higher and adapter to help substantially.

---

## Two-line summary

> Run `bash preflight.sh`, find the path in the decision tree, follow the
> matching 锦囊. End state: ~40 minutes from clone to first WER comparison
> on whichever modern NVIDIA GPU you have.
