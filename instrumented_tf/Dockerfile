# Use Ubuntu 22.04 as a base image for modern toolchains
FROM ubuntu:jammy

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies, Python, and LLVM/Clang
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    wget \
    unzip \
    cpanminus \
    git curl bubblewrap patch make unzip bzip2 opam libgmp-dev python2.7 \
    software-properties-common \
    patchelf \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    clang \
    llvm \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    software-properties-common \
    ninja-build \
    clang \
    llvm-dev \
    libomp-dev \
    libopenblas-dev \
    libblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libssl-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libboost-all-dev \
    lld \
    lcov \
    tmux \
    btop \
    htop \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

ENV CC=clang
ENV CXX=clang++
ENV LDFLAGS="-fuse-ld=lld"

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Install Bazel (version required by TensorFlow 2.16)
ARG BAZEL_VERSION=6.5.0
RUN wget https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    chmod +x bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    ./bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    rm bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh

# Set working directory
WORKDIR /app

# Clone TensorFlow source code
ARG TF_VERSION=v2.16.1
RUN git clone --depth 1 --branch ${TF_VERSION} https://github.com/tensorflow/tensorflow.git

WORKDIR /app/tensorflow

# Install TensorFlow dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install numpy wheel packaging requests opt_einsum auditwheel
RUN python -m pip install keras_nightly

# Configure TensorFlow build
# Set CC and CXX to clang and clang++ for the build
ENV CC=clang
ENV CXX=clang++
# Disable jemalloc to avoid potential build issues
ENV TF_SYSTEM_JEMALLOC=0
# Set Python path for configure script
ENV PYTHON_BIN_PATH=/usr/bin/python
# Accept default configurations non-interactively
ENV TF_CONFIGURE_IOS=0

# Run configure
RUN ./configure

# Build TensorFlow pip package with coverage flags
# -fprofile-instr-generate and -fcoverage-mapping are used for clang-based coverage
RUN bazel build \
    --action_env=CC=clang \
    --action_env=CXX=clang++ \
    --copt="-fprofile-instr-generate" \
    --copt="-fcoverage-mapping" \
    --linkopt="-fprofile-instr-generate" \
    --linkopt="-fcoverage-mapping" \
    --collect_code_coverage \      
    //tensorflow/tools/pip_package:build_pip_package

# Build the pip package
RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg 

# Copy the repository into the Docker image
COPY . /workspace/repo

# Set the working directory to the location of the repo
WORKDIR /workspace/repo

RUN cp /tmp/tensorflow_pkg/tensorflow-*.whl /workspace/repo/instrumented_tf/

RUN auditwheel repair ./instrumented_tf/tensorflow-*.whl

CMD ["bash"]