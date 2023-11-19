# Setup
## Python Version:
Python 3.10. If not possible, delete all type annotations in code and use Python 3.9

## Install Dependencies
```
pip install git+https://github.com/huggingface/transformers.git
pip install sentencepiece
pip install datasets
pip install soundfile
pip install speechbrain
pip install IPython
pip install librosa
pip install soundfile
pip install accelerate -U
pip install tensorboardX
pip install speechbrain
```
Note: if you're using Windows, install `windows-curses` instead of `curses-util`
See `requirements.txt` for complete list.

## Huggingface User Token
User token (with write access) is used to upload the model to huggingface. Use huggingface-cli to login or set `push_to_hub` to `False`

See:
    https://huggingface.co/docs/hub/security-tokens


# How to Use:
1. First, go to `src/constants.py` for settings (whether to train, and training hyperparameters)
2. Run `python3 src/main.py` with commandline parameters under the project root

# Project Structure
- `audio_outputs`: all generated audio will be in this dir
- `data`: processed and cached dataset will live here. Not included in Git
- `log`: all logs from program
- `model`: cached model from previous training. Not included in Git
- `pretrained-models` & `speecht5_tts`: cached pretrained model. Not included in Git
- `speaker_embeddings`: generated speaker embeddings from Mozilla Common Voice 01 dataset. Seperated by accent and gender
- `src`: all python source files


# Dataset
1. Mozilla Common Voice 
It contains at least 15 sub-dataset, we only use 01 for now.
https://huggingface.co/datasets/mozilla-foundation/common_voice_1_0/tree/main

For Mozilla Common Voice 01 English, the gender and accent distribution is the following:

8637 samples total
```
'us male': 4236, 'us female': 454,
'england male': 1614, 'england female': 53,
'australia male': 963, 'australia female': 5,
'canada male': 480, 'canada female': 142, 
'indian male': 319, 'indian female': 9,
'malaysia male': 82,
'african male': 50,
'newzealand male': 43,
'philippines female': 14,
'ireland male': 12, 'ireland female': 21,
'hongkong male': 4, 'hongkong female': 16,
'scotland male': 4,  'wales male': 3,
'other male': 112, 'other female': 1,
```

2. Facebook's Voxpopuli. We currently don't use it since it lacks gender annotation for `en_accented` 
https://huggingface.co/datasets/facebook/voxpopuli/tree/main/data/en_accented

# Evaluation
We use https://huggingface.co/dima806/english_accents_classification for evaluation

# Troubleshoot
1. Manully download dataset using voxpopuli repo
https://github.com/facebookresearch/voxpopuli/issues/43#

2. If you're using Windows/WSL and dataset.map is stucked while using multi-processing
https://discuss.huggingface.co/t/map-multiprocessing-issue/4085/25
