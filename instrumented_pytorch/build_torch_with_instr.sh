####### 4. build 

whl=${1:-1} # generate wheel if 1, build if 0. default 1.

if [ $whl -eq 1 ]; then
    mode=bdist_wheel
else
    mode=develop
fi

cd pytorch
python setup.py clean

export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export CXXFLAGS=""
USE_CPP_CODE_COVERAGE=1 \
USE_KINETO=0 BUILD_CAFFE2=0 USE_DISTRIBUTED=0 USE_NCCL=0 BUILD_TEST=0 USE_XNNPACK=0 \
USE_QNNPACK=0 USE_MIOPEN=0 BUILD_CAFFE2_OPS=0 USE_TENSORPIPE=0 CMAKE_VERBOSE_MAKEFILE=ON \
USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=0 USE_FBGEMM=0 USE_NNPACK=0 USE_GOLD_LINKER=1 CC=clang CXX=clang++ python setup.py ${mode}
