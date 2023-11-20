import socket
import time

###############################################################################
# Training & Dataset Related
###############################################################################
# By default we use the mozilla one, since it contains necessary metadata on speaker gender & accent
# remote_dataset_name = "facebook/voxpopuli"
remote_dataset_name = "mozilla-foundation/common_voice_1_0"
remote_dataset_subset = "en"
remote_dataset_split = "train"
download_remote_dataset = False  # if False, load local dataset
save_processed_dataset = True  # if True, save the processed (prepare_dataset) dataset to disk.

train_model = True  # if False, load saved model/checkpoints
save_fine_tuned_model = True
dataset_train_size = 1000  # can be `int` or `float`. `int` means absolute count; while `float` means percentage
dataset_test_size = 100

###############################################################################
# Hyper-Parameters
###############################################################################
batch_size = 10  # dependent on how much VRAM you have, on my 8G RTX 2070, it should be able to handle a size of 16
gradient_accumulation_steps = 1
learning_rate = 1e-5  # 0.00001
# https://datascience.stackexchange.com/questions/55991/in-the-context-of-deep-learning-what-is-training-warmup-steps
warm_up_step = 500
max_steps = 40000  # Default -1. When set to a positive number, overrides num_train_epochs
save_steps = 10000
eval_steps = 250
logging_steps = 500
###############################################################################
# Files & Directories
###############################################################################
# Model Related
MODEL_BASE_PATH = './model/'
CHECKPOINT_BASE_PATH = "./speecht5_tts/"
DATA_BASE_PATH = "./data/"
EMBEDDINGS_BASE_PATH = './speaker_embeddings/'
AUDIO_OUTPUT_PATH = './audio_outputs/'

model_name = 'trained_' + socket.gethostname() + '_' + time.strftime("%b%e_%H:%M", time.localtime())
model_path = MODEL_BASE_PATH + model_name
data_path = DATA_BASE_PATH
log_file_path = './log/' + socket.gethostname() + '.log'
log_file = open(log_file_path, 'a')

unprocessed_data_path = './raw_data/'

###############################################################################
# HuggingFace Related
###############################################################################
huggingface_token = "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"
huggingface_kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_1_0",
    "dataset": "Common Voice",
    "dataset_args": "config: en",
    "language": "en",
    "model_name": "SpeechT5 TTS English Accented",
    "finetuned_from": "microsoft/speecht5_tts",
    "tasks": "text-to-speech",
    "tags": "en_accent,mozilla,t5,common_voice_1_0",
}
push_to_hub = True

###############################################################################
# Commandline Arguments
###############################################################################
default_gender = 'male'
default_accent = 'england'

###############################################################################
# Others
###############################################################################
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
eval_model_name = 'dima806/english_accents_classification'
all_genders = ['male', 'female']
all_accents = ['us', 'england', 'canada', 'malaysia', 'australia', 'indian', 'hongkong', 'newzealand','philippines']


def log_msg(message, outf=log_file, include_time=True, print_out=True):
    messages = []
    if isinstance(message, str):
        messages.append(message)
    else:
        messages = message

    for m in messages:
        msg = time.strftime("%H:%M:%S", time.localtime()) + '\t' + str(m) if include_time else str(m)
        if print_out: print(msg)
        if outf is not None:
            outf.write(msg)
            outf.write("\n")
            outf.flush()
