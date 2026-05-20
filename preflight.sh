#!/usr/bin/env bash
# Pre-flight check — run this FIRST on the GPU machine.
# Outputs a 6-section diagnostic. Match the output to README.md → "决策树".
#
# Usage:  bash preflight.sh

set +e   # don't bail on first failure — we want the full picture

CHECK="✓"
FAIL="✗"
WARN="!"

echo
echo "========================================"
echo "  meno_docker pre-flight check"
echo "  $(date)  on $(hostname)"
echo "========================================"

echo
echo "------ [1/6] GPU & driver ------"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free \
               --format=csv,noheader 2>/dev/null
    DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    if [ -n "$DRV" ] && [ "$DRV" -ge 555 ] 2>/dev/null; then
        echo "$CHECK  driver $DRV.x >= 555 — Blackwell (RTX 5090) supported"
    elif [ -n "$DRV" ] && [ "$DRV" -ge 525 ] 2>/dev/null; then
        echo "$WARN  driver $DRV.x < 555 — fine for A100/H100, NOT Blackwell"
    else
        echo "$FAIL  driver version unknown or too old"
    fi
else
    echo "$FAIL  nvidia-smi not found — driver not installed?"
fi

echo
echo "------ [2/6] Docker daemon ------"
if command -v docker >/dev/null 2>&1; then
    DV=$(docker --version 2>/dev/null)
    echo "$CHECK  $DV"
    if docker info >/dev/null 2>&1; then
        echo "$CHECK  docker daemon reachable"
        docker info 2>/dev/null | grep -i "runtimes:" | head -1
    else
        echo "$FAIL  docker installed but daemon not running OR no permission"
        echo "      try: sudo systemctl start docker  /  add user to 'docker' group"
    fi
else
    echo "$FAIL  docker NOT installed — ask IT to install docker-ce + nvidia-container-toolkit"
    echo "      (alternative: podman / apptainer — commands translate 1:1)"
fi

echo
echo "------ [3/6] GPU passthrough into container ------"
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    OUT=$(timeout 60 docker run --rm --gpus all \
            nvidia/cuda:12.6.0-base-ubuntu22.04 \
            nvidia-smi --query-gpu=name --format=csv,noheader 2>&1)
    if echo "$OUT" | grep -qi "NVIDIA"; then
        echo "$CHECK  --gpus flag works, container sees:"
        echo "$OUT" | head -8 | sed 's/^/      /'
    else
        echo "$FAIL  container can't see GPU — NVIDIA Container Toolkit missing"
        echo "      first error line:"
        echo "$OUT" | head -3 | sed 's/^/      /'
    fi
else
    echo "$WARN  skipping (no docker)"
fi

echo
echo "------ [4/6] Network egress ------"
for url in https://github.com https://nvcr.io https://huggingface.co https://api.ngc.nvidia.com https://pypi.org; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 6 "$url" 2>/dev/null)
    if [ "$code" = "200" ] || [ "$code" = "301" ] || [ "$code" = "302" ]; then
        echo "$CHECK  $code  $url"
    else
        echo "$FAIL  $code  $url"
    fi
done

echo
echo "------ [5/6] Disk space ------"
DOCKER_ROOT=$(docker info 2>/dev/null | grep "Docker Root Dir" | awk '{print $4}')
if [ -n "$DOCKER_ROOT" ] && [ -d "$DOCKER_ROOT" ]; then
    df -h "$DOCKER_ROOT" 2>/dev/null | tail -1 | awk '{print "  docker root  ("$6") "$4" free"}'
fi
df -h "$HOME" 2>/dev/null | tail -1 | awk '{print "  $HOME        ("$6") "$4" free"}'
df -h "$(pwd)" 2>/dev/null | tail -1 | awk '{print "  workdir      ("$6") "$4" free"}'
echo "  (need ~25 GB on docker root for image + ~5 GB on workdir for outputs)"

echo
echo "------ [6/6] User permissions ------"
if id 2>/dev/null | grep -q '(docker)'; then
    echo "$CHECK  user $(whoami) is in docker group — can run docker without sudo"
else
    echo "$WARN  user $(whoami) NOT in docker group"
    echo "      either: sudo docker ...   OR   sudo usermod -aG docker $(whoami) && logout"
fi

echo
echo "========================================"
echo "  Next:  open README.md → '决策树'  and match this output to a path."
echo "========================================"
