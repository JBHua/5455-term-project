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


# Dataset: Mozilla Common Voice
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

## Filter by Client_ID
To filter and sort using `client_id` on Common Voice 01, we have the following data:
- African female
  - `7e3cb2be1728a33806bf792`
  - 97 samples
- Australia female
  - `d4782952607fb7`
  - 30 samples
- Bermuda female
  - `573685a68e6590c`
  - 3 samples
- Canada female
  - `8b5b5910d4ae5c50`
  - 107
- England female
  - `8ffa59abfe937a6`
  - 247
- Hong Kong female
  - `336fa7c96fefc37`
  - 5
- Indian female
  - `51dff465f51758`
  - 348
- Malaysia female
  - `4acae21d27721648d`
  - 41
- New Zealand female
  - `4bd567a4fd999f1fd2963af8`
  - 20
- Philippines female
  - `e2026b4ea8d3feb81d6faea8e`
  - 21
- Scotland female
  - `38817378c0e5fcf`
  - 4032
- Scotland female
  - `38817378c0e5fcf`
  - 4032
- US female
  - `83986d6998723c149`
  - 322
- Wales female
  - `68b5c286b956`
  - 70
- African male
  - `71e142d5eeaf4d77`
  - 193
- Australia male
  - `7c9901f5b9b0b9f`
  - 417
- Bermuda male
  - `113240cb79b3`
  - 18
- Canada male
  - `756f0d710c13`
  - 262
- England male
  - `44a7b0c7d982bf`
  - 946
- Hong Kong male
  - `6f343eab099660f101`
  - 42
- Indian male
  - `0da876f2b513de3ce276`
  - 569
- Ireland male
  - `8ceed8cbd7d38d0`
  - 40
- Malaysia male
  - `9dfc026b612463dbfce1ef5de`
  - 100
- New Zealand male
  - `fcdb80a519a73e933`
  - 4890
- Philippines male
  - `9ec24631628c749a38d`
  - 114
- Scotland male
  - `63dcd967d7736`
  - 33
- Singapore male
  - `311b2231b73cf56d477f2edb701`
  - 118
- South Atlandtic male
  - `058c198a24d2510616`
  - 50
- US male
  - `e0a33c02ba5ff7aca`
  - 2516
- Wales male
  - `a0766be99a641c86b2c8d2`
  - 156

# Evaluation
We use https://huggingface.co/dima806/english_accents_classification for evaluation

# Troubleshoot
1. Manully download dataset using voxpopuli repo
https://github.com/facebookresearch/voxpopuli/issues/43#

2. If you're using Windows/WSL and dataset.map is stucked while using multi-processing
https://discuss.huggingface.co/t/map-multiprocessing-issue/4085/25
