"""Quick env sanity check — verify torch/cuda + NeMo + adapter API import."""
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "num_gpus", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  gpu[{i}] {torch.cuda.get_device_name(i)}")

import nemo
print("nemo", nemo.__version__)

from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
print("hybrid model class loaded ok")

from nemo.collections.common.parts import adapter_modules
print("adapter_modules ok, has LinearAdapterConfig:",
      hasattr(adapter_modules, "LinearAdapterConfig"))

import jiwer, librosa, soundfile
print("audio/metric deps:", "jiwer", jiwer.__version__,
      "librosa", librosa.__version__)
