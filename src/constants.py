import socket
import time

# Training Related
train_model = False  # if False, load saved model/checkpoints
download_remote_dataset = False  # if False, load local dataset
save_processed_dataset = True  # if True, save the processed (prepare_dataset) dataset to disk.
save_fine_tuned_model = True
load_checkpoint = ''

# Model Related
MODEL_BASE_PATH = './model/'
CHECKPOINT_BASE_PATH = "./speecht5_tts/"
DATA_BASE_PATH = "./data/"

model_path = MODEL_BASE_PATH + 'trained_' + socket.gethostname() + time.strftime("%H:%M:%S", time.localtime())
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
