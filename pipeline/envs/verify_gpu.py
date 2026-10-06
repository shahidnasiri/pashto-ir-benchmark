"""Run this after activating each conda env to confirm GPU visibility
before touching any embedding model. Takes a few seconds."""
import torch

print("=" * 50)
print(f"torch version       : {torch.__version__}")
print(f"CUDA available       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA build            : {torch.version.cuda}")
    print(f"GPU name              : {torch.cuda.get_device_name(0)}")
    print(f"Compute capability    : {torch.cuda.get_device_capability(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM            : {total_vram:.1f} GB")
    # Quick sanity op on GPU
    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print(f"Matmul sanity check   : OK (result shape {tuple(y.shape)})")
else:
    print("GPU NOT visible in this environment — stop and fix before proceeding.")
print("=" * 50)
