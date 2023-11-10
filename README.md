# Setup
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

## Git Large File Storage
Also, you'll need Git Large File Storage to store model checkpoints and complete models.
See: 
    1. https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage?platform=linux
    2. https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage

## Huggingface User Token
User token (with write access) is used to upload the model to huggingface.
See:
    https://huggingface.co/docs/hub/security-tokens

# Dataset
https://huggingface.co/datasets/facebook/voxpopuli/tree/main/data/en_accented


# Troubleshoot
1. Manully download dataset using voxpopuli repo
https://github.com/facebookresearch/voxpopuli/issues/43#

2. If you're using Windows/WSL and dataset.map is stucked while using multi-processing
https://discuss.huggingface.co/t/map-multiprocessing-issue/4085/25
