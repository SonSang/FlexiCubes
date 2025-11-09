conda create --name flexicubes python=3.10 -y
conda activate flexicubes

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.7.0_cu128.html
pip install imageio trimesh tqdm matplotlib ninja
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install git+https://github.com/NVlabs/nvdiffrast/
conda install -c nvidia cuda-toolkit=12.8 -y

# 기본
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CONDA_PREFIX/bin:$PATH"

# ★ 여기 중요: conda CUDA의 실제 헤더/라이브러리 경로
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# (컴파일러가 확실히 보도록 한 번 더)
export CXXFLAGS="-I$CONDA_PREFIX/targets/x86_64-linux/include $CXXFLAGS"

# 5090(Blackwell) = CC 12.0
export TORCH_CUDA_ARCH_LIST="12.0"

rm -rf ~/.cache/torch_extensions/*

python - <<'PY'
import torch, nvdiffrast.torch as dr
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
ctx = dr.RasterizeCudaContext()
print("nvdiffrast CUDA context OK")
PY