"""Check GPU access and the runtime compiler/import path before training."""
import os
import shutil
import subprocess

from .common import configure_gpu


def main():
    configure_gpu()
    compiler = os.environ.get('CC', 'gcc')
    if not shutil.which(compiler):
        raise RuntimeError('C compiler missing: rebuild the image with build-essential installed')
    subprocess.run([compiler, '--version'], check=True)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one accessible CUDA GPU')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('H200 experiments require BF16 support')
    print(torch.cuda.get_device_name(0), flush=True)
    print('GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30, flush=True)
    # Exercise the import that initializes Triton's CUDA driver and compiles its
    # helper. This is required by PEFT even when the base model is not quantized.
    import bitsandbytes
    from peft.tuners.lora.bnb import dispatch_bnb_8bit
    del dispatch_bnb_8bit
    print(f'bitsandbytes {bitsandbytes.__version__}: import OK', flush=True)
    result = (torch.ones(2, 2, device='cuda', dtype=torch.bfloat16) @
              torch.ones(2, 2, device='cuda', dtype=torch.bfloat16))
    torch.cuda.synchronize()
    if not torch.all(result == 2).item():
        raise RuntimeError('CUDA BF16 computation check failed')
    print('GPU, compiler, PEFT/bitsandbytes and BF16 checks passed', flush=True)


if __name__ == '__main__':
    main()
