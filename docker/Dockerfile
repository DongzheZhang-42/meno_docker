# Custom image: NVIDIA NeMo container + NeMo 1.23 (matches the scripts in
# this repo, which were verified against the 1.23 adapter API).
#
# The base image ships NeMo 2.x — the lines below downgrade to 1.23 and
# install the exact dep tree we validated (see env/requirements.lock.txt
# in the meno_zdz repo for the source-of-truth versions).
#
# Build:
#   docker build -t meno-nemo123:latest -f docker/Dockerfile .
#
# Skip this and use NVIDIA's stock image directly if you're fine running
# against NeMo 2.x (run_stock.sh).

FROM nvcr.io/nvidia/nemo:25.04

WORKDIR /workspace

# Build prereqs that NeMo 1.23's youtokentome dep needs
RUN pip install --no-cache-dir Cython 'setuptools<81' wheel

# Wheel-friendly numeric stack (avoids source-building pyarrow)
RUN pip install --no-cache-dir 'numpy<2' 'pyarrow>=14'

# Downgrade to NeMo 1.23 + the exact compatible deps we validated.
# huggingface_hub<0.23 is needed because NeMo 1.23 uses ModelFilter
# (removed in hub 0.23). transformers/tokenizers must match.
RUN pip uninstall -y nemo_toolkit && \
    pip install --no-cache-dir --no-build-isolation 'youtokentome>=1.0.5' && \
    pip install --no-cache-dir 'nemo_toolkit[asr]==1.23.0' \
                                'huggingface_hub==0.22.2' \
                                'transformers==4.38.2' \
                                'tokenizers<0.16' \
                                'datasets>=2.18,<3' \
                                'pytorch-lightning>=2.0.0,<2.3' \
                                jiwer librosa

# Don't COPY the project into the image — mount it at run time with -v
# so edits on the host are visible inside the container.

ENV HF_HOME=/workspace/.cache_hf \
    NEMO_CACHE_DIR=/workspace/.cache_nemo \
    PYTHONDONTWRITEBYTECODE=1

CMD ["bash"]
