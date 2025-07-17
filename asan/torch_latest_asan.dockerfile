# ==================================================================================================
# PyTorch ASan Build Dockerfile
#
# This Dockerfile builds PyTorch from source with AddressSanitizer (ASan) enabled.
# ==================================================================================================

# --- Base Image ---
FROM ubuntu:24.04 AS builder

# --- Environment Configuration ---
# Set non-interactive mode for package installations and define LLVM version
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC
ARG LLVM_VERSION=18

# --- System & Build Dependencies ---
# Install essential build tools, Python, and PyTorch dependencies in a single layer.
# Also, download and install the specified version of Clang/LLVM.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    wget \
    gnupg \
    lsb-release \
    software-properties-common \
    python3 \
    python3-dev \
    python3-pip \
    libopenblas-dev \
    liblapack-dev \
    ninja-build \
    patch \
    # Install LLVM/Clang
    && wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh ${LLVM_VERSION} all \
    && rm llvm.sh \
    && apt-get install -y --no-install-recommends llvm-${LLVM_VERSION}-dev \
    # Clean up apt cache
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# --- Compiler & Linker Setup ---
# Configure system alternatives to use the installed Clang version as the default compiler.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    ln -s /usr/bin/clang-${LLVM_VERSION} /usr/bin/clang && \
    ln -s /usr/bin/clang++-${LLVM_VERSION} /usr/bin/clang++ && \
    ln -s /usr/bin/llvm-config-${LLVM_VERSION} /usr/bin/llvm-config && \
    ln -s /usr/bin/lld-${LLVM_VERSION} /usr/bin/lld && \
    ln -s /usr/bin/llvm-symbolizer-${LLVM_VERSION} /usr/bin/llvm-symbolizer && \
    update-alternatives --install /usr/bin/ld ld $(which lld-${LLVM_VERSION}) 2

# --- Build Environment Variables ---
# Set environment variables required for the PyTorch build, including compiler paths
# and the library path for the ASan runtime.
ENV CC=clang \
    CXX=clang++ \
    LD_LIBRARY_PATH=/usr/lib/llvm-${LLVM_VERSION}/lib/clang/${LLVM_VERSION}/lib/linux/ \
    LD_PRELOAD=/usr/lib/llvm-${LLVM_VERSION}/lib/clang/${LLVM_VERSION}/lib/linux/libclang_rt.asan-x86_64.so \
    LDSHARED="clang --shared" \
    LDFLAGS="-fsanitize=address -shared-libasan" \
    CFLAGS="-fsanitize=address -fno-sanitize-recover=all -shared-libasan -pthread" \
    CXX_FLAGS="-pthread" \
    UBSAN_FLAGS="-fno-sanitize-recover=all" \
    ASAN_OPTIONS=detect_leaks=0:symbolize=1:strict_init_order=true \
    ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer

# --- Set up repo ---
WORKDIR /workspace
# --- Copy the repository into the Docker image ---
COPY . /workspace/repo
RUN pip install -r /workspace/repo/asan/requirements.txt --break-system-packages

# --- PyTorch Source Checkout ---
WORKDIR /workspace
RUN git clone --recursive --depth 1 -b main https://github.com/pytorch/pytorch.git
WORKDIR /workspace/pytorch

# --- Python Dependencies ---
# Install Python packages required by PyTorch.
RUN pip install -r requirements.txt --break-system-packages

# --- Build and Install PyTorch with ASan ---
# Disable Werror for fbgemm and run the installation with ASan enabled.
RUN sed -i 's/-Werror/-Wno-error/g' /workspace/pytorch/third_party/fbgemm/CMakeLists.txt && \
    DEBUG=1 \
    BUILD_TEST=0 \
    USE_MKLDNN=0 \
    USE_OPENMP=0 \
    USE_CUDA=0 \
    USE_NCCL=0 \
    USE_ASAN=1 \
    BUILD_CAFFE2_OPS=0 \
    USE_DISTRIBUTED=0 \
    python setup.py install

# --- Set the working directory to the location of the repo ---
WORKDIR /workspace/repo

# --- Final Container Setup ---
# Set the default command for the container.
CMD ["/bin/bash"]