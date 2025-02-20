# Deepfake Detection Models (df_models)

This repository provides tools and a deep learning model for detecting deepfake audio samples (anti-spoofing). It includes an end-to-end inference pipeline, sample notebooks for inference on custom WAV files, and a Docker container for easy deployment.

## Overview

The repository includes:

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

- **Python Requirements**:  
  `SSL_Anti-spoofing/requirements.txt` lists the essential dependencies for running the project.

## Features

- **Pretrained DF Detection Model**: Uses a model (downloaded automatically in Docker) for anti-spoofing inference.
- **Custom Audio Dataset**: Supports recursive loading and pre-processing of WAV files.
- **Batch Inference & Continuous Saving**: Processes large datasets in batches, saving results after each inference step.
- **Visualization Tools**: Histograms with threshold lines to help interpret DF detection scores.
- **Dockerized Setup**: Quickly set up the environment using Docker to avoid dependency issues.

## Acknowledgements

This project builds upon the work from the [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) repository by TakHemlata et al., which implements the core deepfake detection model described in their paper:

> "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation" (Speaker Odyssey 2022 Workshop)

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (recommended) or a Python environment with the required packages.
- A GPU with CUDA (optional) for faster inference.

### Docker Setup

1. **Build the Docker Image**

   From the repository root, run:

   ```bash
   docker build -f SSL_Anti-spoofing/Dockerfile -t ssl-antispoof .
   ```

2. **Run the Container**

   Launch the container exposing port 8888 for Jupyter Lab:

   ```bash
   sudo docker run -p 8888:8888 ssl-antispoof
   ```

   This command starts Jupyter Lab. Open the provided URL in your browser to interact with the notebook.


1. Start Jupyter Lab from your Docker container (or local environment):

   ```bash
   jupyter lab
   ```

2. Open `SSL_Anti-spoofing/notebooks/SSL-inference.ipynb`.

3. Follow the notebook cells which will:
   - Generate a list of WAV files (or load them from a text file).
   - Define the custom dataset and loader.
   - Load the pretrained DF detection model.
   - Process audio files in batches and save the detection scores.
   - Visualize the score distribution.

**Note:** Make sure to update any file paths (e.g., where your audio files are located) in the notebook if needed.

## Project Structure

## Citation

If you use this work in your research, please cite the original paper:

```bibtex
@inproceedings{tak2022automatic,
  title={Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation},
  author={Tak, Hemlata and Todisco, Massimiliano and Wang, Xin and Jung, Jee-weon and Yamagishi, Junichi and Evans, Nicholas},
  booktitle={The Speaker and Language Recognition Workshop},
  year={2022}
}
```

The original implementation can be found at:  
[https://github.com/TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)

## License

This project is MIT licensed, following the original repository's license.
