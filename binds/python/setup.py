# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. NVFP4EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Build of the NVFP4EffMA PyTorch extension.

    pip install ./binds/python --no-build-isolation        (from the repository root)

The CUDA library is built first with the repository Makefile (parallel, see its header), for:
  NVFP4EFFMA_ARCHS="61 75"          explicit list of compute capabilities, or
  TORCH_CUDA_ARCH_LIST="6.1;7.5"   the usual PyTorch variable, or
  the GPUs visible to PyTorch at build time, or, without any, every supported one (sm_50..sm_90).
Architectures outside sm_50..sm_90 are dropped: those GPUs have native FP4 (or are too old).
Other variables: CUDA_HOME (toolkit), NVFP4EFFMA_CCBIN (host compiler for nvcc), MAX_JOBS,
NVFP4EFFMA_OBJDIR (reuse an object directory with the library already built for these architectures),
NVFP4EFFMA_LOCAL_VERSION (local version label of a release wheel, e.g. cu126torch2.7).
"""
import os
import re
import subprocess
import sys

from setuptools import find_packages, setup

try:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
except ImportError as e:  # build isolation hides the installed torch
    sys.exit(f"NVFP4EffMA needs PyTorch at build time ({e}); install torch first and use "
             "`pip install --no-build-isolation ...`")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("NVFP4EFFMA_ROOT", os.path.normpath(os.path.join(HERE, "..", "..")))
HEADER = os.path.join(ROOT, "include", "nvfp4effma.h")
if not os.path.isfile(HEADER):
    sys.exit(f"NVFP4EffMA sources not found at {ROOT}; build from a repository checkout or set NVFP4EFFMA_ROOT")
VERSION = re.search(r'#define NVFP4EFFMA_VERSION "([^"]+)"', open(HEADER).read()).group(1)
if os.environ.get("NVFP4EFFMA_LOCAL_VERSION"):  # e.g. cu126torch2.7 for release wheels (dev/release.sh)
    VERSION += "+" + os.environ["NVFP4EFFMA_LOCAL_VERSION"]
SUPPORTED = [50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86, 87, 89, 90]
# the license files live at the repository root; wheels must carry them (setuptools only takes
# license files from the package directory)
LICENSE_FILES = ["LICENSE", "LICENSE-EXCEPTION", "NOTICE"]
for f in LICENSE_FILES:
    src = os.path.join(ROOT, f)
    if os.path.isfile(src):
        with open(src) as i, open(os.path.join(HERE, f), "w") as o: o.write(i.read())
# the C header and the device-side codecs go into the package (nvfp4effma.get_include()) for extensions that
# build on it; src/ next to include/ keeps nvfp4effma.cuh's #include "../include/nvfp4effma.h" valid
HEADERS = {"include": ["nvfp4effma.h"], "src": ["nvfp4effma.cuh", "nvfp4effma_tuned.h"]}
for d, names in HEADERS.items():
    os.makedirs(os.path.join(HERE, "nvfp4effma", d), exist_ok=True)
    for f in names:
        with open(os.path.join(ROOT, d, f)) as i, open(os.path.join(HERE, "nvfp4effma", d, f), "w") as o: o.write(i.read())


def target_archs():
    env = os.environ.get("NVFP4EFFMA_ARCHS") or os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env:
        archs = set()
        for tok in re.split(r"[;,\s]+", env.strip()):
            m = re.fullmatch(r"(\d+)\.?(\d)?(\+PTX)?", tok)
            if not m:
                sys.exit(f"cannot parse architecture '{tok}' (expected e.g. 61 or 6.1)")
            archs.add(int(m.group(1) + (m.group(2) or "")))
        source = "NVFP4EFFMA_ARCHS / TORCH_CUDA_ARCH_LIST"
    elif torch.cuda.is_available() and torch.cuda.device_count():
        archs = {10 * a + b for a, b in (torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count()))}
        source = "visible GPUs"
    else:
        archs, source = set(SUPPORTED), "default (no GPU visible)"
    dropped = sorted(a for a in archs if a not in SUPPORTED)
    archs = sorted(a for a in archs if a in SUPPORTED)
    if dropped:
        print(f"NVFP4EffMA: skipping sm_{', sm_'.join(map(str, dropped))} (native FP4 or unsupported)", file=sys.stderr)
    if not archs:
        sys.exit(f"NVFP4EffMA: no supported architecture to build for ({source}: {sorted(dropped)}); "
                 "it targets sm_50..sm_90 -- GPUs with native FP4 should use cuBLASLt / TE")
    print(f"NVFP4EffMA: building for sm_{', sm_'.join(map(str, archs))} ({source})", file=sys.stderr)
    return archs


class Build(BuildExtension):
    def run(self):
        if CUDA_HOME is None:
            sys.exit("NVFP4EffMA: CUDA toolkit not found; set CUDA_HOME")
        # NVFP4EFFMA_OBJDIR: an object directory that already holds the library for these architectures
        # (dev/release.sh builds it once for all wheels); make then only checks that it is up to date
        objdir = os.path.abspath(os.environ.get("NVFP4EFFMA_OBJDIR") or os.path.join(self.build_temp, "libnvfp4effma"))
        cmd = ["make", "-C", ROOT, f"OBJDIR={objdir}", f"ARCHS={' '.join(map(str, target_archs()))}",
               f"CUDA_HOME={CUDA_HOME}", f"{objdir}/libnvfp4effma.a"]
        if os.environ.get("NVFP4EFFMA_CCBIN"):
            cmd.append(f"CCBIN={os.environ['NVFP4EFFMA_CCBIN']}")
        if os.environ.get("MAX_JOBS"):
            cmd.append(f"JOBS={os.environ['MAX_JOBS']}")
        print(" ".join(cmd), file=sys.stderr)
        subprocess.check_call(cmd)
        for ext in self.extensions:
            ext.extra_objects.append(os.path.join(objdir, "libnvfp4effma.a"))
        super().run()


setup(
    name="nvfp4effma",
    version=VERSION,
    description="NVFP4 linear layers, GEMM / GEMV and quantization for NVIDIA GPUs without native FP4 (sm_50..sm_90)",
    long_description=open(os.path.join(HERE, "README.md")).read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["nvfp4effma", "nvfp4effma.*"]),
    package_data={"nvfp4effma": [f"{d}/{f}" for d, names in HEADERS.items() for f in names]},
    ext_modules=[CUDAExtension("nvfp4effma._C", ["csrc/nvfp4effma_torch.cpp"], include_dirs=[os.path.join(ROOT, "include")],
                               extra_compile_args={"cxx": ["-O2"]})],
    cmdclass={"build_ext": Build},
    python_requires=">=3.9",
    license_files=LICENSE_FILES,
    url="https://github.com/srose69/nvfp4effma",
    install_requires=["torch>=2.1"],
    zip_safe=False,
)
