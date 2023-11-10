import socket
import time

# Training Related
train_model = False # if False, load saved model/checkpoints
download_and_process_dataset = False # if False, load local dataset
load_checkpoint = ''

# Model Related
MODEL_BASE_PATH = './model/'
CHECKPOINT_BASE_PATH = "./speecht5_tts/"
DATA_BASE_PATH = "./data/"

model_path = MODEL_BASE_PATH + 'trained_' + socket.gethostname() + time.strftime("%H:%M:%S", time.localtime())
checkpoint_path = CHECKPOINT_BASE_PATH + '/checkpoint-4000'
data_path = DATA_BASE_PATH

# Others
huggingface_token = "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"
log_file_path = './log/' + socket.gethostname() + '.log'
log_file = open(log_file_path, 'a')
