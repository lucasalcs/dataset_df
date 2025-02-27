# Use PyTorch base image matching the repository recommendation
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the AASIST repository
RUN git clone https://github.com/clovaai/aasist.git .

# Upgrade pip first
RUN pip install --upgrade pip

# Copy your requirements file
COPY requirements.txt /app/external_requirements.txt

# Install both repository requirements and external requirements
# Note: Installing external_requirements.txt last ensures it can override any conflicting packages
RUN pip install -r requirements.txt && pip install -r /app/external_requirements.txt

# Copy the inference notebook
COPY notebooks/AASIST-inference.ipynb /app

# Return to the main app directory
WORKDIR /app

# Expose port
EXPOSE 8888

# Use simple jupyter notebook command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]