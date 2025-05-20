# Use PyTorch base image with the required versions
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# Set working directory
WORKDIR /app

# Create dedicated directories
RUN mkdir -p /app/sls_model /app/data /app/outputs

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    unzip \
    libsndfile1 \
    libsndfile1-dev \
    sed \
    && rm -rf /var/lib/apt/lists/*

# Clone the SLSforASVspoof repository into the dedicated directory
WORKDIR /app/sls_model
RUN git clone https://github.com/QiShanZhang/SLSforASVspoof-2021-DF.git .

# Unzip the fairseq package as specified in the installation instructions
RUN unzip fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1.zip

# Install only essential fairseq dependencies without reinstalling PyTorch
WORKDIR /app/sls_model/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
RUN pip install --no-deps hydra-core==1.0.7 "omegaconf<2.1" sacrebleu>=1.4.12 \
    cython bitarray regex antlr4-python3-runtime==4.8 portalocker \
    && pip install -e . --no-deps

# Return to the main app directory
WORKDIR /app

# Copy and install requirements from the project's sls directory
COPY requirements.txt /app/sls_requirements.txt
RUN pip install -r /app/sls_requirements.txt

# Install gdown to download from Google Drive
RUN pip install gdown

# Create directory for pretrained models within the sls_model directory
RUN mkdir -p /app/sls_model/pretrained_models

# Download pretrained model checkpoint from Google Drive into the correct location
RUN gdown "https://drive.google.com/uc?id=1gfyrxFihJevsqVIOO-YS0WY55CuWS_xJ" -O /app/sls_model/pretrained_models/asvdf_sls_best.pth

# Download XLS-R wav2vec model into the correct location
RUN wget --progress=bar:force https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -O /app/sls_model/xlsr2_300m.pt

# Fix hardcoded path to xlsr2_300m.pt in the cloned repository's model.py
# This assumes the original path is 'xlsr2_300m.pt' directly in the script.
# If the actual relative path is different (e.g., './xlsr2_300m.pt') this sed command might need adjustment.
RUN sed -i "s|'xlsr2_300m.pt'|'/app/sls_model/xlsr2_300m.pt'|g" /app/sls_model/model.py

# Copy the inference notebook (adjust path if it's inside the cloned repo or project's notebooks dir)
# Assuming it's in the project's models/sls/notebooks directory for now
COPY notebooks/SLS-inference.ipynb /app/SLS-inference.ipynb

# Add sls_model directory to Python path
RUN echo "/app/sls_model" > /opt/conda/lib/python3.8/site-packages/sls_model.pth

# Expose port for Jupyter
EXPOSE 8888

# Launch Jupyter Lab on container startup
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]
