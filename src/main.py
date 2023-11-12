from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SpeechT5Tokenizer, PreTrainedModel
from collections import defaultdict
from datasets import DatasetDict, Dataset, IterableDatasetDict, IterableDataset
import torch
import matplotlib.pyplot as plt
import os
import sys
import curses
import traceback
from speechbrain.pretrained import EncoderClassifier
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import soundfile as sf
import time
import constants
from datasets import load_dataset, load_from_disk
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

###############################################################################
# Helper Functions
###############################################################################
def log_msg(message, outf=constants.log_file, include_time=True, print_out=True):
    msg = time.strftime("%H:%M:%S", time.localtime()) + '\t' + message if include_time else message
    if print_out: print(msg)
    if outf is not None:
        outf.write(msg)
        outf.write("\n")
        outf.flush()


def error_recording_hook(exctype, value, tb):
    if exctype == KeyboardInterrupt:
        log_msg("KeyboardInterrupt detected. Exiting...")
    else:
        log_msg(f'Exception: {"".join(traceback.format_exception(exctype, value=value, tb=tb))}', print_out=False)
        sys.__excepthook__(exctype, value, tb)


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
        if pretrained_model.config.reduction_factor > 1:
            target_lengths = torch.tensor([
                len(feature["input_values"]) for feature in label_features
            ])
            target_lengths = target_lengths.new([
                length - length % pretrained_model.config.reduction_factor for length in target_lengths
            ])
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch


def init_global_variables() -> tuple[PreTrainedModel, SpeechT5Processor, SpeechT5Tokenizer, EncoderClassifier,
                                     TTSDataCollatorWithPadding, PreTrainedModel]:
    log_msg("Initializing pretrained model, processor, tokenizer, speaker_model, data_collator, and vocoder")

    _pretrained_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    _processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    _tokenizer = _processor.tokenizer
    _speaker_model = EncoderClassifier.from_hparams(
        source=constants.spk_model_name,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", constants.spk_model_name)
    )
    _data_collator = TTSDataCollatorWithPadding(processor=_processor)
    _vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    return _pretrained_model, _processor, _tokenizer, _speaker_model, _data_collator, _vocoder


def select_speaker(speaker_id):
    return 100 <= speaker_counts[speaker_id] <= 400


def create_speaker_embedding(waveform):
    with torch.no_grad():
        _speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        _speaker_embeddings = torch.nn.functional.normalize(_speaker_embeddings, dim=2)
        _speaker_embeddings = _speaker_embeddings.squeeze().cpu().numpy()
    return _speaker_embeddings


def prepare_dataset(example):
    """prepare_dataset takes a single entry; tokenize input text; load audio into a log-mel spectrogram; and add
    speaker embeddings"""
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
    """filters out entry with short text (to prevent T5 from complaining). default cutoff is 10 characters"""
    return len(raw_text) > cutoff


def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200


def list_directories(path):
    """List all subdirectories in a given path"""
    for root, dirs, _ in os.walk(path):
        for dir in dirs:
            yield os.path.join(root, dir)


def print_menu(stdscr, selected_row_idx, directories):
    stdscr.clear()
    h, w = stdscr.getmaxyx()  # Get the height and width of the window

    for idx, row in enumerate(directories):
        if len(row) > w - 2:  # Check if the row is too long and truncate it if necessary
            row = row[:w-5] + '...'

        x = max(0, w//2 - len(row)//2)  # Ensure x is within the window
        y = max(0, h//2 - len(directories)//2 + idx)  # Ensure y is within the window

        if idx == selected_row_idx:
            stdscr.attron(curses.color_pair(1))
            stdscr.addstr(y, x, row)
            stdscr.attroff(curses.color_pair(1))
        else:
            stdscr.addstr(y, x, row)

    stdscr.refresh()


def run_menu(stdscr, dirs):
    """Handles the menu navigation and selection"""
    curses.curs_set(0)  # Turn off cursor blinking
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Color scheme for selected row

    current_row = 0
    print_menu(stdscr, current_row, dirs)

    while True:
        key = stdscr.getch()

        if key == curses.KEY_UP and current_row > 0:
            current_row -= 1
        elif key == curses.KEY_DOWN and current_row < len(dirs) - 1:
            current_row += 1
        elif key == curses.KEY_ENTER or key in [10, 13]:
            stdscr.addstr(0, 0, f"You've selected model path: '{dirs[current_row]}'")
            return dirs[current_row]

        print_menu(stdscr, current_row, dirs)


def select_local_mode():
    directories = list(list_directories(constants.MODEL_BASE_PATH))
    if len(directories) == 0:
        log_msg(f"No model under dir {constants.MODEL_BASE_PATH}. Please double check. Exiting")
        sys.exit()
    elif len(directories) == 1:
        selected_model = directories[0]
        log_msg(f"Using model: {selected_model}")
        return selected_model

    selected_model = curses.wrapper(run_menu, directories)
    log_msg(f"Using model: {selected_model}")
    return selected_model


###############################################################################
# Setup
###############################################################################
sys.excepthook = error_recording_hook

device = "cuda" if torch.cuda.is_available() else "cpu"
cpu_count = len(os.sched_getaffinity(0))
torch.set_num_threads(cpu_count)

log_msg(
    f'\nStart -- {time.strftime("%b %e %H:%M:%S", time.localtime())} -- pytorch={torch.version.__version__}, device={device}, cpu_count={cpu_count}',
    include_time=False)

###############################################################################
# Global Variables
###############################################################################
pretrained_model, processor, tokenizer, speaker_model, data_collator, vocoder = init_global_variables()


###############################################################################
# Load Datasets
###############################################################################
def set_sampling_rate(_dataset):
    from datasets import Audio

    log_msg("Start Setting Sampling Rate to 16 kHz...")
    # SpeechT5 requires the sampling rate to be 16 kHz.
    _dataset = _dataset.cast_column("audio", Audio(sampling_rate=16000))
    log_msg("Setting Sampling Rate Successfully")

    return _dataset


def clean_mozilla_dataset(_dataset):
    def normalize_text(entry):
        entry['sentence'] = entry['sentence'].lower()
        return entry

    log_msg(f"Using Mozilla Common Voice. Additional Cleaning Needed")
    log_msg(f"Starting Size of Mozilla Common Voice: {len(_dataset)}")

    _dataset = _dataset.filter(lambda entry: len(entry["accent"]) > 0)
    log_msg(f"Size after filtering accent: {len(_dataset)}")

    _dataset = _dataset.filter(lambda entry: entry["gender"] in ['female', 'male'])
    log_msg(f"Size after filtering gender: {len(_dataset)}")

    _dataset = _dataset.filter(lambda entry: int(entry['down_votes']) <= int(entry['up_votes']))
    log_msg(f"Size after filtering voting: {len(_dataset)}")

    _dataset = _dataset.map(normalize_text)
    _dataset = _dataset.rename_column("sentence", "normalized_text")
    log_msg(f"Finish normalizing input text")

    log_msg("Finish Cleaning Dataset. Length of dataset: " + str(len(_dataset)))

    return _dataset


def load_remote_dataset(name=constants.remote_dataset_name,
                        subset=constants.remote_dataset_subset) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    log_msg(f"Loading Remote Dataset: {name}. Sub-collection: {subset}")
    _dataset = load_dataset(
        name, subset, split=constants.remote_dataset_split,
        download_mode="reuse_cache_if_exists", num_proc=cpu_count
    )

    log_msg("Finish Loading Remote Dataset. Length of dataset: " + str(len(_dataset)))
    if constants.remote_dataset_name.startswith("mozilla-foundation"):
        # # TODO: Remove next line, we dont need to save dataset now. It's just for speeding up the process of debugging
        # _dataset.save_to_disk(constants.data_path)
        _dataset = clean_mozilla_dataset(_dataset)

    return set_sampling_rate(_dataset)


def load_local_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    log_msg("Using Locally Processed Dataset. Skip Processing...")

    try:
        _dataset = load_from_disk(constants.data_path)
    except Exception as e:
        log_msg(f"Failed to load local dataset. Please double check file exists and path is correct. {e}")
        sys.exit()

    return _dataset


###############################################################################
# Select Speaker
###############################################################################


###############################################################################
# Speaker Embeddings
###############################################################################


###############################################################################
# Process Entire Dataset
###############################################################################
# def sort_speaker(_dataset):
#     print(_dataset)
#
#     speaker_counts = defaultdict(int)
#     for speaker_id in _dataset["speaker_id"]:
#         speaker_counts[speaker_id] += 1
#     plt.figure()
#     plt.hist(speaker_counts.values(), bins=5)
#     plt.ylabel("Speakers")
#     plt.xlabel("Examples")
#     plt.show()
#
#     accent_count = defaultdict(int)
#     for accent_id in _dataset["accent"]:
#         accent_count[accent_id] += 1
#     plt.figure()
#     plt.hist(accent_count.values(), bins=20)
#     plt.ylabel("Accent")
#     plt.xlabel("Examples")
#     plt.show()
#
#     gender_count = defaultdict(int)
#     for g_id in _dataset["gender"]:
#         gender_count[g_id] += 1
#     plt.figure()
#     plt.hist(gender_count.values(), bins=5)
#     plt.ylabel("Gender")
#     plt.xlabel("Examples")
#     plt.show()


def filter_and_prepare_dataset(_dataset) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    log_msg("Start Filtering Short Data")
    _dataset = _dataset.filter(is_not_too_short, input_columns=["normalized_text"])
    log_msg(f"{len(_dataset)} data entries left after filtering")

    log_msg("Start Preparing Dataset")
    _dataset = _dataset.map(prepare_dataset, remove_columns=_dataset.column_names, num_proc=1)
    # _dataset = _dataset.map(DatasetPrepper(processor), remove_columns=_dataset.column_names, num_proc=cpu_count)
    log_msg("Preparing dataset finished successfully")

    log_msg("Start Filtering Long Data")  # TODO: why can't we do it beforehand?
    _dataset = _dataset.filter(is_not_too_long, input_columns=["input_ids"])
    log_msg(f"{len(_dataset)} data entries left after filtering")

    return _dataset


def split_dataset(_dataset, test_size=constants.dataset_test_size, train_size=constants.dataset_train_size):
    _dataset = _dataset.train_test_split(test_size=test_size, train_size=train_size)
    return _dataset


def generate_train_arguments():
    log_msg("Generating Seq2SeqTrainer Arguments")
    return Seq2SeqTrainingArguments(
        output_dir=constants.CHECKPOINT_BASE_PATH,
        per_device_train_batch_size=6,
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
        push_to_hub=constants.push_to_hub,
    )


def generate_trainer(_training_args: Seq2SeqTrainingArguments, _model: PreTrainedModel, _dataset, _data_collator,
                     _tokenizer):
    log_msg("Generating Seq2SeqTrainer")
    return Seq2SeqTrainer(
        args=_training_args,
        model=_model,
        train_dataset=_dataset["train"],
        eval_dataset=_dataset["test"],
        data_collator=_data_collator,
        tokenizer=_tokenizer,
    )


if __name__ == "__main__":
    # Step 1: Download & Split datasets
    dataset = None
    if constants.download_remote_dataset:
        dataset = load_remote_dataset()
        # Step 1A: Process & Preparing Dataset
        dataset = filter_and_prepare_dataset(dataset)

        if constants.save_processed_dataset:
            log_msg(f"save_processed_dataset is True. Saving processed dataset to dir: {constants.data_path}")
            dataset.save_to_disk(constants.data_path)
    else:
        # data in local file is guaranteed to be processed. So we don't need to process it again
        dataset = load_local_dataset()

    # Step 2: Split Dataset
    divided_dataset = split_dataset(dataset)

    # Step 3: Train or Load the model
    if constants.train_model:
        pretrained_model.config.use_cache = False
        training_args = generate_train_arguments()
        trainer = generate_trainer(training_args, pretrained_model, divided_dataset, data_collator, tokenizer)

        log_msg("Start Training...")
        trainer.train()
        log_msg("Start Saving the model...")

        if constants.save_fine_tuned_model: trainer.save_model(constants.model_path)
        if constants.push_to_hub: trainer.push_to_hub(**constants.huggingface_kwargs)
    else:
        pretrained_model = SpeechT5ForTextToSpeech.from_pretrained(pretrained_model_name_or_path=select_local_mode(),
                                                                   local_files_only=True).to(device)

    text = "I'm loading the model from the Hugging Face Hub!"
    log_msg(f'Input text: {text}')
    inputs = processor(text=text, return_tensors="pt").to(device)

    example = divided_dataset["train"][5]
    speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0).to(device)
    spectrogram = pretrained_model.generate_speech(inputs["input_ids"], speaker_embeddings).to(device)

    with torch.no_grad():
        speech = vocoder(spectrogram)
    speech = speech.cpu()  # move back to CPU

    from IPython.display import Audio
    Audio(speech.numpy(), rate=16000)

    import soundfile as sf
    sf.write("output.wav", speech.numpy(), samplerate=16000)
