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

**2)** Visit [Pytorch](https://pytorch.org/) and follow the installation guide for **GPU** packages

**3)** Once installed, inside ```src/hts-audio-transformer/audio_detection.py``` under ```class Audio_Classification``` change ```self.device = torch.device('cpu')``` to ```self.device = torch.device('cuda')```

## Model Artifacts
Two files will be necessary in order for the script to run:

**1) AudioSet checkpoint**

  -Main checkpoint can be found in the SurvAI's google cloud, this checkpoint has shown to be the most consistant.
  
  -For other AudioSet checkpoints, visit the following Google Drive [link](https://drive.google.com/drive/folders/1cZhMO7qLXTeifXVPP7PdM1NRYCG5cx28).
  
**2) Class hierachy map**

-```class_hier_map.npy``` is found within the model_artifacts folder contained inside this repo, class map contains every class provided by the AudioSet dataset. This will be utilized for performing classifaction on each audio file. 



## Inference

The main script is ```run_audio.py``` this will take a folder of videos, process them and perform audio detection.



This script takes the following arguments:

### --folder
Provide the path to your video(s) folder to which you would like to detect audio from, argument accepts nested directories. (defaulted to '')

### --save-output 
If argument is included, JSON will be saved to "output_files". (defaulted to False)


#### Example for running local inference:
``` python src/run_audio.py --folder video --save-output```



## Reference
Codebase has been cleaned of some unrelated files and code has been modified. For complete model repository visit RetroCirce's github: [HTSAT](https://github.com/RetroCirce/HTS-Audio-Transformer)
