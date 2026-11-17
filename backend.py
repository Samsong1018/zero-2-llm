# Picks the array library for tensor.py and nn.py, once per process.
#
# CPU (NumPy) is the default, so every existing project behaves exactly as before.
# To run a script on the GPU, set the switch BEFORE importing tensor / nn:
#
#     os.environ["MLP_DEVICE"] = "gpu"
#     from tensor import Tensor
#
# or from the shell:  MLP_DEVICE=gpu python3 GPT/gpt_shakespear.py

import os
import numpy

DEVICE = os.environ.get("MLP_DEVICE", "cpu").lower()

if DEVICE == "gpu":
    try:
        import cupy as xp
        import cupyx
    except ImportError as e:
        raise ImportError("MLP_DEVICE=gpu but CuPy isn't installed (pip install cupy-cuda13x)") from e

    def scatter_add(a, idx, values):
        cupyx.scatter_add(a, idx, values)

    def to_numpy(a):
        return xp.asnumpy(a)

elif DEVICE == "cpu":
    xp = numpy

    def scatter_add(a, idx, values):
        numpy.add.at(a, idx, values)

    def to_numpy(a):
        return a

else:
    raise ValueError(f"MLP_DEVICE must be 'cpu' or 'gpu', got {DEVICE!r}")
