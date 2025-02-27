# Deepfake Detection Models (df_models)

This repository provides tools and deep learning models for detecting deepfake audio samples (anti-spoofing). It includes end-to-end inference pipelines, sample notebooks for inference on custom WAV files, and Docker containers for easy deployment of two state-of-the-art deepfake detection systems: SSL_Anti-spoofing and AASIST.

## Overview

The repository includes:

- **SSL_Anti-spoofing Model**:
  - **Inference Notebook**:  
    `SSL_Anti-spoofing/notebooks/SSL-inference.ipynb` provides step-by-step instructions for:
    - Loading and pre-processing your audio files.
    - Running batch inference using a pretrained deepfake (DF) detection model.
    - Visualizing the distribution of detection scores with threshold annotations.

  - **Docker Container**:  
    `SSL_Anti-spoofing/Dockerfile` builds a containerized environment with:
    - A PyTorch runtime image.
    - All necessary system dependencies and Python libraries.
    - Automatic downloads of the pretrained model (Best_LA_model_for_DF.pth) and XLS-R wav2vec model.
    - A pre-configured Jupyter Lab server ready to launch on container startup.

- **AASIST Model**:
  - **Inference Notebook**:  
    `AASIST/notebooks/AASIST-inference.ipynb` offers similar capabilities for the AASIST model:
    - Processing audio files using AASIST's graph attention networks.
    - Running inference with pre-trained models.
    - Visualizing results with the EER threshold.

  - **Docker Container**:  
    `AASIST/Dockerfile` provides a ready-to-use environment with:
    - A PyTorch 1.6.0 runtime image with CUDA 10.1 support.
    - The complete AASIST codebase and dependencies.
    - Pre-trained AASIST and AASIST-L models.
    - A Jupyter Lab interface for running the notebook.

- **Python Requirements**:  
  Each model folder contains its own `requirements.txt` file listing the essential dependencies.

## Features

- **Multiple Detection Models**: 
  - **SSL_Anti-spoofing**: Uses self-supervised learning with wav2vec 2.0 for anti-spoofing.
  - **AASIST**: Uses integrated spectro-temporal graph attention networks.
- **Custom Audio Dataset**: Supports recursive loading and pre-processing of WAV/FLAC files.
- **Batch Inference & Continuous Saving**: Processes large datasets in batches, saving results after each inference step.
- **Visualization Tools**: Histograms with threshold lines to help interpret detection scores.
- **Dockerized Setup**: Quickly set up each environment using Docker to avoid dependency issues.

## Acknowledgements

This project builds upon the work from two repositories:

1. [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) by TakHemlata et al., which implements a deepfake detection model using wav2vec 2.0.

2. [AASIST](https://github.com/clovaai/aasist) by NAVER Corporation, which implements audio anti-spoofing using integrated spectro-temporal graph attention networks.

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (recommended) or a Python environment with the required packages.
- A GPU with CUDA (optional) for faster inference.

### Docker Setup

1. **Build the Docker Images**

   From the repository root, run one of the following to build your preferred model:

   ```bash
   # For SSL_Anti-spoofing
   docker build -f SSL_Anti-spoofing/Dockerfile -t ssl-antispoof .
   
   # For AASIST
   docker build -f AASIST/Dockerfile -t aasist .
   ```

2. **Run the Container**

   Launch the container exposing port 8888 for Jupyter Lab:

   ```bash
   # For SSL_Anti-spoofing
   docker run -p 8888:8888 ssl-antispoof
   
   # For AASIST
   docker run -p 8888:8888 aasist
   ```

   This command starts Jupyter Lab. Open the provided URL in your browser to interact with the notebook.

### Using the Notebooks

1. Start Jupyter Lab from your Docker container (or local environment)
2. Open either `SSL_Anti-spoofing/notebooks/SSL-inference.ipynb` or `AASIST/notebooks/AASIST-inference.ipynb`
3. Follow the notebook cells which will:
   - Generate a list of audio files (or load them from a text file).
   - Define the custom dataset and loader.
   - Load the pretrained detection model.
   - Process audio files in batches and save the detection scores.
   - Visualize the score distribution.

**Note:** Make sure to update any file paths (e.g., where your audio files are located) in the notebook if needed.

## Citation

If you use this work in your research, please cite the original papers:

For SSL_Anti-spoofing:
```bibtex
@inproceedings{tak2022automatic,
  title={Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation},
  author={Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle={The Speaker and Language Recognition Workshop},
  year={2022}
}
```

For AASIST:
```bibtex
@INPROCEEDINGS{Jung2021AASIST,
  author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks}, 
  year={2022}
}
```

The original implementations can be found at:  
- SSL_Anti-spoofing: [https://github.com/TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
- AASIST: [https://github.com/clovaai/aasist](https://github.com/clovaai/aasist)

## License

This project is MIT licensed, following the original repositories' licenses.
