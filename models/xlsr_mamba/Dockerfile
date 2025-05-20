# Use PyTorch devel base image matching requirements.txt (PyTorch 2.2.1, CUDA 11.8)
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Set frontend to noninteractive to avoid prompts during apt-get install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (including download tools)
# Add grep and sed for the download workaround
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    libsndfile1 \
    libsndfile1-dev \
    unzip \
    grep \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install gdown, then downgrade pip to avoid metadata issues
# gdown no longer needed for model zip, but keep for potential future use
RUN pip install --upgrade pip && pip install gdown && pip install pip==24.0

# --- Download stage --- 
# Clone repositories first
RUN git clone https://github.com/swagshaw/XLSR-Mamba.git .
RUN git clone https://github.com/facebookresearch/fairseq.git fairseq_dir

# Download only the base XLSR model directly to /app
# RUN mkdir -p /app/pretrained_models && \ # Removed: No longer needed
#    # Download the base XLSR model (fine-tuned models will be mounted)
RUN echo "Downloading base XLSR model..." && \
    wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -O /app/xlsr2_300m.pt && \
    echo "Base XLSR download complete."

# --- Installation stage --- 
# Install fairseq at the commit required by the authors
WORKDIR /app/fairseq_dir 
RUN git checkout a54021305d6b3c
# Patch fairseq code for compatibility with newer NumPy AFTER checkout
RUN sed -i 's/np.float/np.float64/g' fairseq/data/indexed_dataset.py && \
    pip install --editable ./

# Install repository and notebook requirements
WORKDIR /app
# Comment out gpuRIR and remove torch/torchvision/torchaudio AND nvidia-* packages from requirements.txt
RUN sed -i -e '/^gpuRIR/s/^/#/' -e '/^torch==/d' -e '/^torchvision==/d' -e '/^torchaudio==/d' -e '/^nvidia-/d' requirements.txt
RUN pip install -r requirements.txt
COPY notebook_requirements.txt /app/notebook_requirements.txt
RUN pip install -r notebook_requirements.txt

# Copy the inference notebook into the image
COPY notebooks/XLSR-Mamba-inference.ipynb /app/XLSR-Mamba-inference.ipynb

# --- Runtime configuration --- 
# Expose Jupyter port
EXPOSE 8888

# Default command launches Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"] 