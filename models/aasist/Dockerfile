# Use PyTorch base image matching the repository recommendation
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Set the main application directory
WORKDIR /app

# Create organized directory structure
RUN mkdir -p /app/aasist_code /app/data /app/outputs

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    wget \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone the AASIST repository into the dedicated directory
WORKDIR /app/aasist_code
RUN git clone https://github.com/clovaai/aasist.git .
# The AASIST repo comes with pre-trained models in 'models/anti-spoofing_model_g_TD_last.pth' etc.
# and 'models/pretrained_anti-spoofing_model_using_various_vocoders/AASIST-L.pth'
# These will be available at /app/aasist_code/models/

# Upgrade pip first
RUN pip install --upgrade pip

# Copy external requirements file (if any specific to your setup, otherwise use repo's)
# Assuming the requirements.txt in models/aasist/ is the one to use
COPY requirements.txt /app/aasist_code/external_requirements.txt

# Install repository requirements and then external requirements
WORKDIR /app/aasist_code
RUN pip install -r requirements.txt
RUN pip install -r external_requirements.txt # If this contains additional/overriding packages

# Return to the main app directory for the notebook and Jupyter Lab
WORKDIR /app

# Copy the inference notebook to the root of the Jupyter environment
COPY notebooks/AASIST-inference.ipynb /app/AASIST-inference.ipynb

# Make Python aware of modules in /app/aasist_code for the notebook
# AASIST structure seems to be runnable scripts, but if it has importable modules:
# RUN echo "/app/aasist_code" > /usr/local/lib/python3.6/dist-packages/aasist_custom.pth
# For PyTorch 1.6.0 (Python 3.6/3.7/3.8 typically), path might vary.
# A more robust way if site-packages path is uncertain:
# RUN python -c "import site; print(site.getsitepackages()[0])" # to find path
# Let's assume for now direct script execution or that AASIST handles its path within its scripts.
# If model.py or similar needs to be imported, a .pth file will be good.

# Force install compatible version of markupsafe for newer Jupyter versions
RUN pip install "markupsafe<2.1"

# Fix for traitlets/ipykernel compatibility issue if needed for Jupyter Lab
RUN pip install "traitlets>=5.0,<5.10" "ipykernel>=6.0,<6.26" "jupyter_client>=7.0,<7.5" "jupyter_server>=1.0,<1.19"

# Expose port
EXPOSE 8888

# Use Jupyter Lab
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]