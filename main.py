from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, Audio
import torch
import soundfile as sf
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import soundfile as sf

###############################################################################
# Helper Functions
###############################################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'pytorch={torch.version.__version__}, device={device}')

def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

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
tokenizer = processor.tokenizer

# inputs = processor(text="Don't count the days, make the days count.", return_tensors="pt")

###############################################################################
# Load Datasets
###############################################################################
from datasets import load_dataset, Audio
print("Loading Dataset...")
dataset = load_dataset(
    # "facebook/voxpopuli", "en_accented", split="test"
    "facebook/voxpopuli", "en", split="test[:15]",
    download_mode="reuse_cache_if_exists"
)
print("Finish Loading Dataset")

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# print("Length of dataset: " + str(len(dataset)))
# vocabs = dataset.map(
#     extract_all_chars, 
#     batched=True, 
#     batch_size=-1, 
#     keep_in_memory=True, 
#     remove_columns=dataset.column_names,
# )
# dataset_vocab = set(vocabs["vocab"][0])
# tokenizer_vocab = {k for k,_ in tokenizer.get_vocab().items()}

###############################################################################
# Select Speaker
###############################################################################
speaker_counts = defaultdict(int)

plt.figure()
plt.hist(speaker_counts.values(), bins=20)
plt.ylabel("Speakers")
plt.xlabel("Examples")
plt.show()

for speaker_id in dataset["speaker_id"]:
    speaker_counts[speaker_id] += 1
dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])

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
plt.figure()
plt.imshow(processed_example["labels"].T)
plt.show()

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
dataset = dataset.map(
    prepare_dataset, remove_columns=dataset.column_names,
)
dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
print(len(dataset)) # How many examples left?

dataset = dataset.train_test_split(test_size=0.1)
print(dataset)

data_collator = TTSDataCollatorWithPadding(processor=processor)
features = [
    dataset["train"][0],
    dataset["train"][1],
    dataset["train"][20],
]

batch = data_collator(features)
{k:v.shape for k,v in batch.items()}
# sf.write("tts_example.wav", speech.numpy(), samplerate=16000)

model.config.use_cache = True


training_args = Seq2SeqTrainingArguments(
    output_dir="./speecht5_tts",  # change to a repo name of your choice
    per_device_train_batch_size=16,
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

trainer.train()

###############################################################################
# Process Entire Dataset
###############################################################################
model = SpeechT5ForTextToSpeech.from_pretrained("Matthijs/speecht5_tts")

example = dataset["test"][304]
speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
print("speaker_embeddings.shape")
print(speaker_embeddings.shape)

text = "hello, this is pytorch!"
inputs = processor(text=text, return_tensors="pt")
spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
plt.figure()
plt.imshow(spectrogram.T)
plt.show()

with torch.no_grad():
    speech = vocoder(spectrogram)

from IPython.display import Audio
Audio(speech.numpy(), rate=16000)

sf.write("output.wav", speech.numpy(), samplerate=16000)