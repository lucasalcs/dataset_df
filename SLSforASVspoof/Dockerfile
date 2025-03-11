# Use PyTorch base image with the required versions
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    unzip \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the SLSforASVspoof repository
RUN git clone https://github.com/QiShanZhang/SLSforASVspoof-2021-DF.git .

# Unzip the fairseq package as specified in the installation instructions
RUN unzip fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip

# Install only essential fairseq dependencies without reinstalling PyTorch
# This avoids the "No space left on device" error
WORKDIR /app/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
RUN pip install --no-deps hydra-core==1.0.7 "omegaconf<2.1" sacrebleu>=1.4.12 \
    cython bitarray regex antlr4-python3-runtime==4.8 portalocker \
    && pip install -e . --no-deps

# Return to the main directory and install requirements
WORKDIR /app
RUN pip install -r requirements.txt

# Install additional dependencies for notebook functionality
COPY requirements.txt /app/external_requirements.txt
RUN pip install -r /app/external_requirements.txt

# Install gdown to download from Google Drive
RUN pip install gdown

# Create directory for pretrained models
RUN mkdir -p /app/pretrained_models

# Download pretrained model checkpoint from Google Drive
RUN gdown "https://drive.google.com/uc?id=1gfyrxFihJevsqVIOO-YS0WY55CuWS_xJ" -O /app/pretrained_models/asvdf_sls_best.pth

# Download XLS-R wav2vec model
RUN wget --progress=bar:force https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -O /app/xlsr2_300m.pt

# Copy the inference notebook - assuming this is within the SLSforASVspoof directory
COPY notebooks/SLS-inference.ipynb /app

# Expose port for Jupyter
EXPOSE 8888

# Launch Jupyter Lab on container startup
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
