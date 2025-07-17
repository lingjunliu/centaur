# ==================================================================================================
# TensorFlow ASan Build Dockerfile
#
# This Dockerfile builds TensorFlow from source with AddressSanitizer (ASan) enabled.
# ==================================================================================================

# --- Base Image ---
FROM ubuntu:24.04 AS builder

# --- Environment Configuration ---
# Set non-interactive mode for package installations and define versions.
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC
ARG LLVM_VERSION=19
ARG TENSORFLOW_VERSION=v2.19.0
ARG BAZEL_VERSION=6.5.0

# --- System & Build Dependencies ---
# Install essential build tools, Python, and Clang/LLVM in a single layer.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    ca-certificates \
    cmake \
    git \
    fzf \
    tmux \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    wget \
    unzip \
    curl \
    patchelf && \
    # Install LLVM/Clang
    wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh ${LLVM_VERSION} && \
    rm llvm.sh && \
    apt-get install -y --no-install-recommends \
    libfuzzer-${LLVM_VERSION}-dev \
    clang-${LLVM_VERSION} \
    lld-${LLVM_VERSION} \
    libclang-${LLVM_VERSION}-dev \
    llvm-${LLVM_VERSION} \
    llvm-${LLVM_VERSION}-dev \
    libc++-${LLVM_VERSION}-dev \
    libc++abi-${LLVM_VERSION}-dev \
    clangd-${LLVM_VERSION} \
    clang-format-${LLVM_VERSION} \
    libclang-rt-${LLVM_VERSION}-dev \
    libasan6 && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- Compiler & Linker Setup ---
# Configure system alternatives to use the installed Clang version as the default compiler.
RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/ld ld /usr/bin/lld-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-${LLVM_VERSION} 100 && \
    update-alternatives --install /usr/bin/llvm-symbolizer llvm-symbolizer /usr/bin/llvm-symbolizer-${LLVM_VERSION} 100

# --- Bazel Installation ---
RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-linux-x86_64 -O /usr/local/bin/bazel && \
    chmod +x /usr/local/bin/bazel

# --- Set up repo ---
WORKDIR /workspace
# --- Copy the repository into the Docker image ---
COPY . /workspace/repo
RUN pip install -r /workspace/repo/asan/requirements_tf.txt --break-system-packages

# --- Python Dependencies ---
RUN pip install --no-cache-dir -U \
    "numpy==2.1.1" \
    packaging \
    "protobuf==4.25.3" --break-system-packages

# --- TensorFlow Source Checkout & Patching ---
WORKDIR /
RUN git clone --branch ${TENSORFLOW_VERSION} https://github.com/tensorflow/tensorflow.git /tensorflow --depth 1
WORKDIR /tensorflow
COPY tensorflow-2.18.0-clang19-compat.patch /tensorflow
RUN git apply /tensorflow/tensorflow-2.18.0-clang19-compat.patch && \
    rm /tensorflow/tensorflow-2.18.0-clang19-compat.patch

# --- TensorFlow Build Configuration ---
# Set environment variables for a CPU-only build with Clang and ASan.
ENV PYTHON_BIN_PATH=/usr/bin/python3 \
    USE_DEFAULT_PYTHON_LIB_PATH=1 \
    TF_NEED_ROCM=0 \
    TF_NEED_CUDA=0 \
    TF_NEED_CLANG=1 \
    CC_OPT_FLAGS=-Wno-sign-compare \
    TF_SET_ANDROID_WORKSPACE=0 \
    CC=clang \
    CXX=clang++ \
    LD_PRELOAD=/usr/lib/llvm-${LLVM_VERSION}/lib/clang/${LLVM_VERSION}/lib/linux/libclang_rt.asan-x86_64.so \
    ASAN_OPTIONS=detect_leaks=0:symbolize=1:detect_odr_violation=0  \
    ASAN_SYMBOLIZER_PATH=/usr/bin/llvm-symbolizer \
    LD_LIBRARY_PATH=/usr/lib/llvm-${LLVM_VERSION}/lib/clang/${LLVM_VERSION}/lib/linux/ 

# --- Build and Install TensorFlow ---
# Configure, build the wheel, and install it.
RUN ./configure && \
    bazel build \
    --copt="-fsanitize=address" \
    --copt="-Wno-c23-extensions" \
    --cxxopt="-fsanitize=address" \
    --cxxopt="-Wno-c23-extensions" \
    --linkopt="-fsanitize=address" \
    --linkopt=-shared-libasan \
    //tensorflow/tools/pip_package:wheel && \
    pip install bazel-bin/tensorflow/tools/pip_package/wheel_house/tensorflow-*.whl --break-system-packages

# --- Set the working directory to the location of the repo ---
WORKDIR /workspace/repo

# --- Final Container Setup ---
# Set the default command for the container.
CMD ["bash"]