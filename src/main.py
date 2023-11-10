from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
import torch
import matplotlib.pyplot as plt
import os
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import soundfile as sf
import time
import constants
from datasets import load_dataset

###############################################################################
# Helper Functions
###############################################################################
def log_msg(message, outf=constants.log_file):
    msg = time.strftime("%H:%M:%S", time.localtime()) + '\t' + message
    print(msg)
    if not outf is None:
        outf.write(msg)
        outf.write("\n")
        outf.flush()


def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400


def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings


def prepare_dataset(example):
    """prepare_dataset takes a single entry; tokenize input text; load audio into a log-mel spectrogram; and add speaker embeddings"""
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


def is_not_too_short(raw_text, cutoff: int = 10):
    """is_not_too_short filters out entry with short text (to prevent T5 from complaining). default cutoff is 10 characters"""
    return len(raw_text) > cutoff


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
# Setup
###############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
cpu_count = len(os.sched_getaffinity(0))
torch.set_num_threads(cpu_count)

log_msg(f'Start -- {time.strftime("%b %e %H:%M:%S", time.localtime())} -- pytorch={torch.version.__version__}, device={device}, cpu_count={cpu_count}')

###############################################################################
# Load Datasets
###############################################################################
def load_remote_dataset(name="facebook/voxpopuli", subset = "en_accented") -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    log_msg(f"Loading Remote Dataset: {name}. Sub-collection: {subset}")
    dataset = load_dataset(
        name, subset, split="test",
        download_mode="reuse_cache_if_exists",
        keep_in_memory=True, num_proc=cpu_count
    )
    log_msg("Finish Loading Remote Dataset. Length of dataset:" + str(len(dataset)))

    log_msg("Start Setting Sampling Rate to 16 kHz...")
    from datasets import Audio
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000)) # SpeechT5 requires the sampling rate to be 16 kHz.
    log_msg("Setting Sampling Rate Successfully")
    
    return dataset


def load_local_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    log_msg("Using Locally Processed Dataset. Skip Processing...")
    dataset = load_dataset(constants.data_path)

    return dataset

# dataset = dataset.train_test_split(test_size=5,train_size=15, shuffle=False).values()
# print(dataset)
# print("Finish Spliting Dataset")


###############################################################################
# Select Speaker
###############################################################################
from collections import defaultdict
speaker_counts = defaultdict(int)

## plt.figure()
## plt.hist(speaker_counts.values(), bins=20)
## plt.ylabel("Speakers")
## plt.xlabel("Examples")
## plt.show()

## for speaker_id in dataset["speaker_id"]:
##     speaker_counts[speaker_id] += 1
## dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])

###############################################################################
# Speaker Embeddings
###############################################################################
# spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
# speaker_model = EncoderClassifier.from_hparams(
#     source=spk_model_name, 
#     run_opts={"device": device}, 
#     savedir=os.path.join("/tmp", spk_model_name)
# )
# processed_example = prepare_dataset(dataset[0])
# print(list(processed_example.keys())) # ['input_ids', 'labels', 'speaker_embeddings']
# tokenizer.decode(processed_example["input_ids"])

# print(processed_example["speaker_embeddings"].shape) # (512,)

## plt.figure()
## plt.imshow(processed_example["labels"].T)
## plt.show()
## print(processed_example["speaker_embeddings"].shape) # (512,)
## plt.figure()
## plt.imshow(processed_example["labels"].T)
## plt.show()

# vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# spectrogram = torch.tensor(processed_example["labels"])

## embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
## speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

###############################################################################
# Training
###############################################################################
# with torch.no_grad():
#     speech = vocoder(spectrogram)

# from IPython.display import Audio
# Audio(speech.cpu().numpy(), rate=16000)

###############################################################################
# Process Entire Dataset
###############################################################################
# print("Start preparing dataset...")
# print("Current Dataset Columns:")
## print(dataset.column_names)
## print(type(dataset))
## print(dataset)

def filter_and_prepare_dataset(dataset):
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
    return dataset

def split_dataset(dataset, test_size=5, train_size=15):
    dataset = dataset.train_test_split(test_size=test_size, train_size=train_size)
    return dataset

# data_collator = TTSDataCollatorWithPadding(processor=processor)
# features = [
#     dataset["train"][0],
#     dataset["train"][1],
#     dataset["train"][5],
# ]

# batch = data_collator(features)
# {k:v.shape for k,v in batch.items()}
# sf.write("tts_example.wav", speech.numpy(), samplerate=16000)

# model.config.use_cache = True

# training_args = Seq2SeqTrainingArguments(
#     output_dir="./speecht5_tts",  # change to a repo name of your choice
#     per_device_train_batch_size=32,
#     gradient_accumulation_steps=2,
#     learning_rate=1e-5,
#     warmup_steps=500,
#     max_steps=4000,
#     gradient_checkpointing=True,
#     fp16=True,
#     evaluation_strategy="steps",
#     per_device_eval_batch_size=8,
#     save_steps=1000,
#     eval_steps=1000,
#     logging_steps=25,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     greater_is_better=False,
#     label_names=["labels"],
#     push_to_hub=False,
# )

# trainer = Seq2SeqTrainer(
#     args=training_args,
#     model=model,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     data_collator=data_collator,
#     tokenizer=processor.tokenizer,
# )

# dataset.save_to_disk("data_path")

# if constants.train:
#     trainer.train()
#     trainer.save_model(constants.model_path)

#     kwargs = {
#         "dataset_tags": "facebook/voxpopuli",
#         "dataset": "VoxPopuli",  # a 'pretty' name for the training dataset
#         "dataset_args": "config: nl, split: train",
#         "language": "en_accent",
#         "model_name": "SpeechT5 English Accent",  # a 'pretty' name for your model
#         "finetuned_from": "microsoft/speecht5_tts",
#         "tasks": "text-to-speech",
#         "tags": "",
#     }
#     trainer.push_to_hub(**kwargs)
# else:
#     model = SpeechT5ForTextToSpeech.from_pretrained(pretrained_model_name_or_path=constants.checkpoint_path, local_files_only=True)    

###############################################################################
# Process Entire Dataset
###############################################################################

# # model = SpeechT5ForTextToSpeech.from_pretrained(pretrained_model_name_or_path=model_path, local_files_only=True)

# example = dataset["test"][2]
# speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
# print("speaker_embeddings.shape")
# print(speaker_embeddings.shape)

# text = "hello, this is pytorch!"
# inputs = processor(text=text, return_tensors="pt")
# spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
# # plt.figure()
# # plt.imshow(spectrogram.T)
# # plt.show()

# with torch.no_grad():
#     speech = vocoder(spectrogram)

# from IPython.display import Audio
# Audio(speech.numpy(), rate=16000)

# sf.write("output.wav", speech.numpy(), samplerate=16000)

if __name__ == "__main__":
    # Step 1: Download & Split datasets
    dataset = None
    if constants.download_and_process_dataset:
        dataset = load_remote_dataset()
    else:
        dataset = load_local_dataset()

    # Step 2: Load Pre-trained Models
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    tokenizer = processor.tokenizer
    
