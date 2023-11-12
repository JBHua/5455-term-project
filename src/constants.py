import socket
import time

# Training Related
# By default we use the mozilla one, since it contains necessary metadata on speaker gender & accent
# remote_dataset_name = "facebook/voxpopuli"
remote_dataset_name = "mozilla-foundation/common_voice_1_0"
remote_dataset_subset = "en"
remote_dataset_split = "train"
download_remote_dataset = False  # if False, load local dataset
save_processed_dataset = True  # if True, save the processed (prepare_dataset) dataset to disk.

train_model = False  # if False, load saved model/checkpoints
save_fine_tuned_model = True
dataset_train_size = 25  # can be `int` or `float`. `int` means absolute count; while `float` means percentage
dataset_test_size = 5

# Model Related
MODEL_BASE_PATH = './model/'
CHECKPOINT_BASE_PATH = "./speecht5_tts/"
DATA_BASE_PATH = "./data/"

model_path = MODEL_BASE_PATH + 'trained_' + socket.gethostname() + '_' + time.strftime("%b%e_%H:%M", time.localtime())
checkpoint_path = CHECKPOINT_BASE_PATH + '/checkpoint-4000'
data_path = DATA_BASE_PATH
log_file_path = './log/' + socket.gethostname() + '.log'
log_file = open(log_file_path, 'a')

# HuggingFace Related
huggingface_token = "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"
huggingface_kwargs = {
    "dataset_tags": "facebook/voxpopuli",
    "dataset": "VoxPopuli",
    "dataset_args": "config: en_accented",
    "language": "en",
    "model_name": "SpeechT5 TTS English Accented",
    "finetuned_from": "microsoft/speecht5_tts",
    "tasks": "text-to-speech",
    "tags": "en_accent,vocpopuli,t5",
}
push_to_hub = True

# Others
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
