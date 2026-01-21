# VEGA Reproduction Environment
# Ubuntu 24.04 with CUDA 11.7 support for VEGA AI-driven compiler backend generation

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

LABEL maintainer="VEGA Reproduction"
LABEL description="Isolated environment for VEGA compiler backend generation research"

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set up locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    nano \
    htop \
    unzip \
    zip \
    tar \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda
RUN conda init bash && \
    echo "conda activate vega_ae" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Clone VEGA repository
RUN git lfs install && \
    git lfs clone https://huggingface.co/docz1105/VEGA_AE /workspace/VEGA_AE || \
    git clone https://huggingface.co/docz1105/VEGA_AE /workspace/VEGA_AE

# Create conda environment for VEGA
RUN conda create -n vega_ae python=3.8.1 -y

# Activate environment and install dependencies
SHELL ["conda", "run", "-n", "vega_ae", "/bin/bash", "-c"]

# Install PyTorch with CUDA 11.7 support
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
RUN pip install transformers==4.25.1 \
    numpy==1.24.1 \
    tqdm==4.64.1 \
    tree-sitter==0.20.1 \
    tensorboardX==2.5.1 \
    scikit-learn==1.2.0

# Copy requirements if exists in VEGA_AE
RUN if [ -f /workspace/VEGA_AE/requirements.txt ]; then pip install -r /workspace/VEGA_AE/requirements.txt; fi

# Create directory for test code
RUN mkdir -p /workspace/tests

# Set default shell back
SHELL ["/bin/bash", "-c"]

# Environment variables
ENV PYTHONPATH="/workspace/VEGA_AE:${PYTHONPATH}"

# Entry point
CMD ["/bin/bash"]
