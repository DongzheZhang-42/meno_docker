"""Inspect ConformerEncoderAdapter to verify it's a structurally-identical subclass
of ConformerEncoder (so we can safely swap __class__ at runtime)."""
from nemo.collections.asr.modules import ConformerEncoder, ConformerEncoderAdapter
from nemo.core.classes.mixins import adapter_mixins
import inspect

print("ConformerEncoderAdapter bases:", ConformerEncoderAdapter.__bases__)
print("issubclass of ConformerEncoder:", issubclass(ConformerEncoderAdapter, ConformerEncoder))
print("has AdapterModuleMixin in MRO:",
      adapter_mixins.AdapterModuleMixin in ConformerEncoderAdapter.__mro__)
print()
print("ConformerEncoderAdapter.__init__ signature:")
print(" ", inspect.signature(ConformerEncoderAdapter.__init__))
print()
# Show only methods added by the adapter subclass
parent_methods = set(dir(ConformerEncoder))
new_methods = [n for n in dir(ConformerEncoderAdapter)
               if n not in parent_methods and not n.startswith("_")]
print("New methods in ConformerEncoderAdapter:", new_methods)

# Also check the layer
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
print("\nConformerLayer adapter-mixin?",
      adapter_mixins.AdapterModuleMixin in ConformerLayer.__mro__)
