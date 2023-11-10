from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Audio
import torch
import soundfile as sf
import matplotlib.pyplot as plt
import os
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import soundfile as sf
import multiprocess
import socket
import time

huggingface_token = "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"
###############################################################################
# Helper Functions
###############################################################################
model_path = './model/trained_' + socket.gethostname() + time.strftime("%H:%M:%S", time.localtime())
checkpoint_path = './speecht5_tts/checkpoint-4000'
train=False
data_path = './dataset/' + 'english_accented.hf'
# multiprocess.set_start_method("spawn", force=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'pytorch={torch.version.__version__}, device={device}')
cpu_count = len(os.sched_getaffinity(0))
print("Using " + str(cpu_count) + " cpu")
torch.set_num_threads(cpu_count)


def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

def prepare_dataset(example):
    # load the audio data; if necessary, this resamples the audio to 16kHz
    audio = example["audio"]

    # feature extraction and tokenization
    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"], 
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    example["labels"] = example["labels"][0]

    # use SpeechBrain to obtain x-vector
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

    return example

def is_not_too_short(raw_text):
    return len(raw_text) > 10

def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )        

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor([
                len(feature["input_values"]) for feature in label_features
            ])
            target_lengths = target_lengths.new([
                length - length % model.config.reduction_factor for length in target_lengths
            ])
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch

###############################################################################
# Load Pre-trained Models
###############################################################################
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

# inputs = processor(text="Don't count the days, make the days count.", return_tensors="pt")

###############################################################################
# Load Datasets
###############################################################################
from datasets import load_dataset, Audio
print("Loading Dataset...")
dataset = load_dataset(
    # "facebook/voxpopuli", "en_accented", split="test"
    "facebook/voxpopuli", "en_accented", split="test",
    download_mode="reuse_cache_if_exists",
    keep_in_memory=True,
    num_proc=cpu_count
)
print("Finish Loading Dataset... Length of dataset:" + str(len(dataset)))

print("Start Setting Sampling Rate...")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# dataset = dataset.train_test_split(test_size=5,train_size=15, shuffle=False).values()
# print(dataset)
# print("Finish Spliting Dataset")

tokenizer = processor.tokenizer

###############################################################################
# Select Speaker
###############################################################################
from collections import defaultdict
speaker_counts = defaultdict(int)

# plt.figure()
# plt.hist(speaker_counts.values(), bins=20)
# plt.ylabel("Speakers")
# plt.xlabel("Examples")
# plt.show()

# for speaker_id in dataset["speaker_id"]:
#     speaker_counts[speaker_id] += 1
# dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])

###############################################################################
# Speaker Embeddings
###############################################################################
spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name, 
    run_opts={"device": device}, 
    savedir=os.path.join("/tmp", spk_model_name)
)
processed_example = prepare_dataset(dataset[0])
print(list(processed_example.keys())) # ['input_ids', 'labels', 'speaker_embeddings']
tokenizer.decode(processed_example["input_ids"])

print(processed_example["speaker_embeddings"].shape) # (512,)
# plt.figure()
# plt.imshow(processed_example["labels"].T)
# plt.show()
# print(processed_example["speaker_embeddings"].shape) # (512,)
# plt.figure()
# plt.imshow(processed_example["labels"].T)
# plt.show()

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
spectrogram = torch.tensor(processed_example["labels"])
# embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

###############################################################################
# Training
###############################################################################
with torch.no_grad():
    speech = vocoder(spectrogram)

from IPython.display import Audio
Audio(speech.cpu().numpy(), rate=16000)

###############################################################################
# Process Entire Dataset
###############################################################################
print("Start preparing dataset...")
print("Current Dataset Columns:")
print(dataset.column_names)
print(type(dataset))
print(dataset)

dataset = dataset.filter(is_not_too_short, input_columns=["normalized_text"])
print("How many examples left after filtering is_not_too_short?")
print(len(dataset)) # How many examples left?

dataset = dataset.map(
    prepare_dataset, remove_columns=dataset.column_names,
)

print("Preparing dataset finished succesfully")
print(dataset)
dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
print("How many examples left after filtering not_too_long?")
print(len(dataset)) # How many examples left?

dataset = dataset.train_test_split(test_size=5, train_size=15)
print(dataset)

data_collator = TTSDataCollatorWithPadding(processor=processor)
features = [
    dataset["train"][0],
    dataset["train"][1],
    dataset["train"][5],
]

batch = data_collator(features)
{k:v.shape for k,v in batch.items()}
sf.write("tts_example.wav", speech.numpy(), samplerate=16000)

model.config.use_cache = True

training_args = Seq2SeqTrainingArguments(
    output_dir="./speecht5_tts",  # change to a repo name of your choice
    per_device_train_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    greater_is_better=False,
    label_names=["labels"],
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)

dataset.save_to_disk("data_path")

if train:
    trainer.train()
    trainer.save_model(model_path)

    kwargs = {
        "dataset_tags": "facebook/voxpopuli",
        "dataset": "VoxPopuli",  # a 'pretty' name for the training dataset
        "dataset_args": "config: nl, split: train",
        "language": "en_accent",
        "model_name": "SpeechT5 English Accent",  # a 'pretty' name for your model
        "finetuned_from": "microsoft/speecht5_tts",
        "tasks": "text-to-speech",
        "tags": "",
    }
    trainer.push_to_hub(**kwargs)
else:
    model = SpeechT5ForTextToSpeech.from_pretrained(pretrained_model_name_or_path=checkpoint_path, local_files_only=True)    

###############################################################################
# Process Entire Dataset
###############################################################################

# model = SpeechT5ForTextToSpeech.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)

example = dataset["test"][2]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
print("speaker_embeddings.shape")
print(speaker_embeddings.shape)

text = "hello, this is pytorch!"
inputs = processor(text=text, return_tensors="pt")
spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
# plt.figure()
# plt.imshow(spectrogram.T)
# plt.show()

with torch.no_grad():
    speech = vocoder(spectrogram)

from IPython.display import Audio
Audio(speech.numpy(), rate=16000)

sf.write("output.wav", speech.numpy(), samplerate=16000)

