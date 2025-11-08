# # 0) Activate your Spack env (the one where you installed cuda@12.9, gcc@14, etc.)
# spack env activate llama-cpp-cuda
# spack load cuda@12.9 gcc@14 cmake@3.28: ninja ccache

# # 1) Point to the Spack installs
# CUDA_PREFIX="$(spack location -i cuda@12.9)"
# GCC_BIN_DIR="$(spack location -i gcc@14)/bin"

# # 2) Clean the build cache so CMake forgets the old nvcc
# rm -rf llama.cpp/build

# # 3) Configure, *pinning* both the CUDA compiler and the host compiler to Spack:
# cmake -S llama.cpp -B llama.cpp/build \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DBUILD_SHARED_LIBS=ON \
#   -DGGML_CUDA=ON \
#   -DLLAMA_CURL=ON \
#   -DCMAKE_CUDA_ARCHITECTURES=120 \
#   -DCMAKE_CUDA_COMPILER="${CUDA_PREFIX}/bin/nvcc" \
#   -DCMAKE_CUDA_HOST_COMPILER="${GCC_BIN_DIR}/g++" \
#   -DCMAKE_C_COMPILER="${GCC_BIN_DIR}/gcc" \
#   -DCMAKE_CXX_COMPILER="${GCC_BIN_DIR}/g++"

# # 4) Build
# cmake --build llama.cpp/build -j


./llama.cpp/llama-server     -hf unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q4_K_XL     --jinja -ngl 99 --threads -1 --ctx-size 80000     --temp 0.7 --min-p 0.0 --top-p 0.80 --top-k 20 --repeat-penalty 1.05
