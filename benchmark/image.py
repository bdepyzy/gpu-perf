import modal

PYTHON_VERSION = "3.13"
CUTLASS_DSL_VERSION = "4.7.0"
B200_GPU = "B200"
B200_TIMEOUT = 30 * 60

b200_base_image = modal.Image.debian_slim(python_version=PYTHON_VERSION).uv_pip_install("torch==2.11.0", f"nvidia-cutlass-dsl[cu13]=={CUTLASS_DSL_VERSION}", "numpy==2.5.2", "einops==0.8.2")
