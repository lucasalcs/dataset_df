# Changed the base image to the working PyTorch runtime image.
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory.
WORKDIR /app

# Create organized directory structure
RUN mkdir -p /app/data /app/outputs /app/SSL_Anti-spoofing

# Install only the necessary system dependencies.
RUN apt-get update && apt-get install -y git build-essential wget libsndfile1 libsndfile1-dev sed && rm -rf /var/lib/apt/lists/*

# Clone the SSL_Anti-spoofing repository into its own directory
WORKDIR /app/SSL_Anti-spoofing
RUN git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git .

# Modify model.py to use the correct absolute path for xlsr2_300m.pt
RUN sed -i "s|cp_path = 'xlsr2_300m.pt'|cp_path = '/app/SSL_Anti-spoofing/xlsr2_300m.pt'|" model.py

# Change directory into the fairseq folder that comes with the repository.
WORKDIR /app/SSL_Anti-spoofing/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1

# Install fairseq in editable mode.
RUN pip install --editable ./

# Return to the repository root.
WORKDIR /app/SSL_Anti-spoofing

# Install gdown for Google Drive downloads
RUN pip install gdown==5.2.0

# Download Best_LA_model_for_DF.pth from Google Drive and XLS-R wav2vec model early to cache these steps.
RUN gdown "https://drive.google.com/uc?id=1JHBClArVdM-Cr1b8In1iakTV_TvC3HvG" -O /app/SSL_Anti-spoofing/Best_LA_model_for_DF.pth
RUN wget --progress=bar:force https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -O /app/SSL_Anti-spoofing/xlsr2_300m.pt

# Now install the requirements after the large downloads.
RUN pip install -r requirements.txt

# Copy external requirements file and install them.
COPY requirements.txt /app/SSL_Anti-spoofing/external_requirements.txt
RUN pip install -r /app/SSL_Anti-spoofing/external_requirements.txt

# Return to the main app directory for the notebook
WORKDIR /app

# Copies local notebooks folder into container's working directory (at root level for easy access)
COPY notebooks/SSL-inference.ipynb /app/SSL-inference.ipynb

# Instead of symbolic links, make Python aware of module paths - add a .pth file to site-packages
RUN echo "/app" > /opt/conda/lib/python3.8/site-packages/ssl_model.pth && \
    echo "/app/SSL_Anti-spoofing" >> /opt/conda/lib/python3.8/site-packages/ssl_model.pth

# Force install compatible version of markupsafe
RUN pip install "markupsafe<2.1"

# Fix for traitlets/ipykernel compatibility issue
RUN pip install traitlets==5.9.0 ipykernel==6.25.0

# Fix for numpy/numba compatibility
RUN pip install "numpy>=1.22.0"

# Install soundfile and its dependencies
RUN pip install soundfile

# Startup command to launch Jupyter Lab
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Launch the container with a bash shell.
#CMD ["/bin/bash"]
