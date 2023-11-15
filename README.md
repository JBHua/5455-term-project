# Setup
## Python Version:
Python 3.10. If not possible, delete all type annotations in code and use Python 3.9 is ok

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
```


## Huggingface User Token
User token (with write access) is used to upload the model to huggingface.
See:
    https://huggingface.co/docs/hub/security-tokens

# Dataset
https://huggingface.co/datasets/facebook/voxpopuli/tree/main/data/en_accented


https://huggingface.co/datasets/mozilla-foundation/common_voice_1_0/tree/main
For Mozilla Common Voice 01 English, the gender and accent distribution is the following:
8637 samples total
{'us female': 454, 'us male': 4236, 'england male': 1614, 'canada male': 480, 'malaysia male': 82, 'canada female': 142, 'australia male': 963, 'indian male': 319, 'hongkong female': 16, 'other male': 112, 'hongkong male': 4, 'england female': 53, 'newzealand male': 43, 'philippines female': 14, 'african male': 50, 'indian female': 9, 'ireland female': 21, 'australia female': 5, 'scotland male': 4, 'ireland male': 12, 'wales male': 3, 'other female': 1}

# Troubleshoot
1. Manully download dataset using voxpopuli repo
https://github.com/facebookresearch/voxpopuli/issues/43#

2. If you're using Windows/WSL and dataset.map is stucked while using multi-processing
https://discuss.huggingface.co/t/map-multiprocessing-issue/4085/25
