# Deepfake Detection Models (df_models)

This repository provides tools and deep learning models for detecting deepfake audio samples (anti-spoofing). It includes end-to-end inference pipelines, sample notebooks for inference on custom WAV files, and Docker containers for easy deployment of four state-of-the-art deepfake detection systems: SSL_Anti-spoofing, AASIST, SLS, and XLSR-Mamba.

## Overview

The repository includes:

- **SSL_Anti-spoofing Model**:
  - **Inference Notebook**:  
    `models/ssl_antispoof/notebooks/SSL-inference.ipynb` provides step-by-step instructions for:
    - Loading and pre-processing your audio files.
    - Running batch inference using a pretrained deepfake (DF) detection model.
    - Visualizing the distribution of detection scores with threshold annotations.

  - **Docker Container**:  
    `models/ssl_antispoof/Dockerfile` builds a containerized environment with:
    - A PyTorch runtime image.
    - All necessary system dependencies and Python libraries.
    - Automatic downloads of the pretrained model (Best_LA_model_for_DF.pth) and XLS-R wav2vec model.
    - A pre-configured Jupyter Lab server ready to launch on container startup.

- **AASIST Model**:
  - **Inference Notebook**:  
    `models/aasist/notebooks/AASIST-inference.ipynb` offers similar capabilities for the AASIST model:
    - Processing audio files using AASIST's graph attention networks.
    - Running inference with pre-trained models.
    - Visualizing results with the EER threshold.

  - **Docker Container**:  
    `models/aasist/Dockerfile` provides a ready-to-use environment with:
    - A PyTorch 1.6.0 runtime image with CUDA 10.1 support.
    - The complete AASIST codebase and dependencies.
    - Pre-trained AASIST and AASIST-L models.
    - A Jupyter Lab interface for running the notebook.

- **SLS Model**:
  - **Inference Notebook**:  
    `models/sls/notebooks/SLS-inference.ipynb` provides capabilities for the SLS model:
    - Processing audio files using Supervised Label Smoothing approach.
    - Running inference with the pre-trained model for deepfake detection.
    - Visualizing detection scores with threshold annotations.

  - **Docker Container**:  
    `models/sls/Dockerfile` builds an environment with:
    - A PyTorch 1.12.1 runtime image with CUDA 11.3 support.
    - All required dependencies for the SLS model.
    - Pre-trained model weights from the ASVspoof 2021-DF challenge.
    - A Jupyter Lab interface for running the notebook.

- **XLSR-Mamba Model**:
  - **Description**: Utilizes a dual-column bidirectional state space model combined with self-supervised learning (pre-trained wav2vec 2.0) for spoofing attack detection.
  - **Docker Container**:
    `models/xlsr_mamba/Dockerfile` builds a containerized environment that:
    - Clones the XLSR-Mamba repository (`https://github.com/swagshaw/XLSR-Mamba.git`).
    - Installs dependencies including a specific commit of `fairseq`.
    - Sets up a Jupyter Lab server.
  - **Inference Notebook**:
    An inference notebook may be available within the cloned XLSR-Mamba repository or may need to be adapted/created based on the examples provided in that repository.
  - **Requirements**:
    The primary dependencies are managed within the `Dockerfile`, which installs requirements from the cloned XLSR-Mamba repository and an additional `models/xlsr_mamba/requirements.txt` (if present and copied as `external_requirements.txt`).
  - **Status**: The integration of this model is currently in progress and not yet fully implemented or tested within this project.

- **Python Requirements**:  
  Each model folder (`models/ssl_antispoof`, `models/aasist`, `models/sls`, and potentially `models/xlsr_mamba`) contains its own `requirements.txt` file listing the essential dependencies (or they are handled within the Dockerfile, as with XLSR-Mamba).

## Features

- **Multiple Detection Models**: 
  - **SSL_Anti-spoofing**: Uses self-supervised learning with wav2vec 2.0 for anti-spoofing.
  - **AASIST**: Uses integrated spectro-temporal graph attention networks.
  - **SLSforASVspoof**: Uses supervised label smoothing approach for deepfake detection.
  - **XLSR-Mamba**: Employs a dual-column bidirectional state space model with wav2vec 2.0.
- **Custom Audio Dataset**: Supports recursive loading and pre-processing of WAV/FLAC files.
- **Batch Inference & Continuous Saving**: Processes large datasets in batches, saving results after each inference step.
- **Visualization Tools**: Histograms with threshold lines to help interpret detection scores.
- **Dockerized Setup**: Quickly set up each environment using Docker to avoid dependency issues.

## Acknowledgements

This project builds upon the work from three repositories:

1. [SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing) by TakHemlata et al., which implements a deepfake detection model using wav2vec 2.0.

2. [AASIST](https://github.com/clovaai/aasist) by NAVER Corporation, which implements audio anti-spoofing using integrated spectro-temporal graph attention networks.

3. [SLSforASVspoof-2021-DF](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF) by QiShan Zhang et al., which uses supervised label smoothing for deepfake detection.

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (recommended) or a Python environment with the required packages.
- A GPU with CUDA (optional) for faster inference.

### Docker Setup

1. **Build the Docker Images**

   Navigate to each model's directory (e.g., `models/ssl_antispoof`) and build the Docker image:

   ```bash
   # For SSL_Anti-spoofing
   cd models/ssl_antispoof
   docker build -t ssl-antispoof .
   cd ../.. # Return to project root
   
   # For AASIST
   cd models/aasist
   docker build -t aasist .
   cd ../.. # Return to project root
   
   # For SLS
   cd models/sls
   docker build -t sls-df .
   cd ../.. # Return to project root

   # For XLSR-Mamba
   cd models/xlsr_mamba
   docker build -t xlsr-mamba-df .
   cd ../.. # Return to project root
   ```

   Alternatively, to build without changing directories, specify full paths:
   
   ```bash
   # From repository root
   docker build -f models/ssl_antispoof/Dockerfile -t ssl-antispoof models/ssl_antispoof
   docker build -f models/aasist/Dockerfile -t aasist models/aasist
   docker build -f models/sls/Dockerfile -t sls-df models/sls
   docker build -f models/xlsr_mamba/Dockerfile -t xlsr-mamba-df models/xlsr_mamba
   ```

2. **Run the Container**

   Launch the container exposing port 8888 for Jupyter Lab:

   ```bash
   # For SSL_Anti-spoofing
   docker run -p 8888:8888 ssl-antispoof
   
   # For AASIST
   docker run -p 8888:8888 aasist
   
   # For SLS
   docker run -p 8888:8888 sls-df

   # For XLSR-Mamba
   docker run -p 8889:8888 xlsr-mamba-df # Use a different host port if 8888 is taken
   ```

   **Run with GPU support (recommended for faster inference):**
   
   ```bash
   # For SSL_Anti-spoofing with GPU
   docker run --gpus all -p 8888:8888 ssl-antispoof
   
   # For AASIST with GPU
   docker run --gpus all -p 8888:8888 aasist
   
   # For SLS with GPU
   docker run --gpus all -p 8888:8888 sls-df

   # For XLSR-Mamba with GPU
   docker run --gpus all -p 8889:8888 xlsr-mamba-df # Use a different host port
   ```

   Note: To use GPU acceleration, you must have the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) installed on your host system.

   This command starts Jupyter Lab. Open the provided URL in your browser to interact with the notebook.

3. **Mount Data Directories**

   To analyze your own audio files, mount your data directory to the container:

   ```bash
   docker run --gpus all -p 8888:8888 -v /path/to/your/audio/files:/data/audio_files sls-df
   ```

   For XLSR-Mamba, if you intend to use local data, ensure the cloned repository path inside the container (`/app`) is accessible for data or mount your data to a suitable path like `/data/audio_files` and adjust notebook paths accordingly:
   ```bash
   docker run --gpus all -p 8889:8888 -v /path/to/your/audio/files:/data/audio_files xlsr-mamba-df
   ```

### Using the Notebooks

1. Start Jupyter Lab from your Docker container (or local environment)
2. Open one of the following notebooks:
   - `models/ssl_antispoof/notebooks/SSL-inference.ipynb`  
   - `models/aasist/notebooks/AASIST-inference.ipynb`
   - `models/sls/notebooks/SLS-inference.ipynb`
   - For XLSR-Mamba, navigate to `/app` (or the cloned repo path) within the Jupyter Lab interface to find its notebooks.
3. Follow the notebook cells which will:
   - Generate a list of audio files (or load them from a text file).
   - Define the custom dataset and loader.
   - Load the pretrained detection model.
   - Process audio files in batches and save the detection scores.
   - Visualize the score distribution.

**Note:** Make sure to update any file paths (e.g., where your audio files are located) in the notebook if needed.

## Evaluation Tools

The `deepfake_eval` directory contains Python modules such as `metrics.py` and `plotting.py` which can be used for evaluation tasks.

Additionally, the `examples/evaluation/` directory contains example notebooks for evaluating specific models on the `brspeech_df` dataset:
- `examples/evaluation/aasist_eval_on_brspeech_df.ipynb`
- `examples/evaluation/sls_eval_on_brspeech_df.ipynb`
- `examples/evaluation/ssl_eval_on_brspeech_df.ipynb`

These notebooks demonstrate how to:
- Load model scores and corresponding labels.
- Calculate metrics such as EER (Equal Error Rate).
- Visualize score distributions and error rates.

Users can adapt these examples or use the modules in `deepfake_eval` to build custom evaluation pipelines for their deepfake detection models. The core evaluation logic previously found in `eval_metrics.ipynb` and `df_detection_evaluation.ipynb` can be replicated or extended using these resources.

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

For SLSforASVspoof:
```bibtex
@inproceedings{zhang2021one,
  title={One-class learning towards generalized voice spoofing detection},
  author={Zhang, Qi-Shan and Wang, Hong and Tao, Jian-Hua},
  booktitle={2021 12th International Symposium on Chinese Spoken Language Processing (ISCSLP)},
  pages={1--5},
  year={2021},
  organization={IEEE}
}
```

For XLSR-Mamba:
```bibtex
@article{xiao2024xlsr,
  title={XLSR-Mamba: A Dual-Column Bidirectional State Space Model for Spoofing Attack Detection},
  author={Xiao, Yang and Das, Rohan Kumar},
  journal={arXiv preprint arXiv:2401.00390},
  year={2024}
}
```
(Note: The XLSR-Mamba citation has been updated with the arXiv ID. Please verify and update if a more formal publication reference becomes available.)

The original implementations can be found at:  
- SSL_Anti-spoofing: [https://github.com/TakHemlata/SSL_Anti-spoofing](https://github.com/TakHemlata/SSL_Anti-spoofing)
- AASIST: [https://github.com/clovaai/aasist](https://github.com/clovaai/aasist)
- SLSforASVspoof: [https://github.com/QiShanZhang/SLSforASVspoof-2021-DF](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF)
- XLSR-Mamba: [https://github.com/swagshaw/XLSR-Mamba](https://github.com/swagshaw/XLSR-Mamba)

## License

This project is MIT licensed, following the original repositories' licenses.
