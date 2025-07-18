####### 4. build 

whl=${1:-1} # generate wheel if 1, build if 0. default 1.

if [ $whl -eq 1 ]; then
    mode=bdist_wheel
else
    mode=install
fi

cd pytorch
python setup.py clean

# export CMAKE_POLICY_VERSION_MINIMUM=3.5
# export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CXX=clang++
export CC=clang
USE_CPP_CODE_COVERAGE=1 _GLIBCXX_USE_CXX11_ABI=1 \
CMAKE_CXX_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
CMAKE_C_FLAGS="-fprofile-instr-generate -fcoverage-mapping" \
USE_KINETO=0 BUILD_CAFFE2=0 USE_DISTRIBUTED=0 USE_NCCL=0 BUILD_TEST=0 USE_XNNPACK=0 \
USE_QNNPACK=0 USE_MIOPEN=0 BUILD_CAFFE2_OPS=0 USE_TENSORPIPE=0 CMAKE_VERBOSE_MAKEFILE=ON \
USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0 USE_FBGEMM=0 USE_NNPACK=0 CC=clang CXX=clang++ python setup.py ${mode}

if [ $whl -eq 1 ]; then
    # Copying the whl file
    cp pytorch/dist/torch-* .
fi