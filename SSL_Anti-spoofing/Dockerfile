# Changed the base image to the working PyTorch runtime image.
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory.
WORKDIR /app

# Install only the necessary system dependencies.
RUN apt-get update && apt-get install -y git build-essential wget && rm -rf /var/lib/apt/lists/*

# Clone the SSL_Anti-spoofing repository into /app.
RUN git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git .

# Change directory into the fairseq folder that comes with the repository.
WORKDIR /app/fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1

# Install fairseq in editable mode.
RUN pip install --editable ./

# Return to the repository root to install additional requirements.
WORKDIR /app
RUN pip install -r requirements.txt

# NEW SECTION: Copy external requirements file and install them.
COPY requirements.txt /app/external_requirements.txt
RUN pip install -r /app/external_requirements.txt

# Copies local notebooks folder into container's working directory
COPY notebooks/SSL-inference.ipynb /app/SSL-inference.ipynb

# NEW SECTION: Download Best_LA_model_for_DF.pth from Google Drive.
RUN gdown "https://drive.google.com/uc?id=1JHBClArVdM-Cr1b8In1iakTV_TvC3HvG" -O /app/Best_LA_model_for_DF.pth

# NEW SECTION: Download XLS-R wav2vec model
RUN wget --progress=bar:force https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt -O /app/xlsr2_300m.pt

# NEW SECTION: Force install compatible version of markupsafe
RUN pip install "markupsafe<2.1"

# Startup command to launch Jupyter Lab
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Launch the container with a bash shell.
#CMD ["/bin/bash"]
