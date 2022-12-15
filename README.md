# AUDIO DETECTION REPOSITORY

## Introduction
This repository contains all necessary components for identifying sounds within video recordings. This codebase utilizes the **Hierarchical Token Semantic Audio Transformer (HSTAT)** and is trained on the AudioSet dataset. Detections are created by extracting audio from video files and is split by n seconds, in order to perform classification on each audio split. Data will exported to web-ready JSON, formatted with all significant information needed.


## Setup

**1)** In the root directory of **audio-detector** run the following commands for installation of required packages:

```conda create env -n "your_env_name" python=3.10.4```

```conda activate your_env_name```

```pip install -r requirements.txt```


```
sudo apt install sox
conda install -c conda-forge ffmpeg
```


**NOTICE:** Pytorch packages are installed as **CPU ONLY** versions inside of requirements.txt 

if you would like to utilize **GPU**:

**1)** ```pip uninstall torch torchvision torchaudio```

**2)** Visit [Pytorch](https://pytorch.org/) and follow the installation guide for **GPU** usage

**3)** Once installed, inside ```audio_detection.py``` in ```class Audio_Classification:``` change ```self.device = torch.device('cpu')``` to ```self.device = torch.device('cuda')```


## Inference

The main script is ```run_audio.py``` this will take a folder of videos and export the results to a JSON file inside the generated folder "output_files"

This script takes the following arguments:


