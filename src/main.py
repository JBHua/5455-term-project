from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SpeechT5Tokenizer, PreTrainedModel, pipeline
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
import time
import constants
from datasets import load_dataset, load_from_disk
from collections import defaultdict
import argparse
import json
import itertools
import pyarrow as pa
import pyarrow.compute as compute
import datasets
from collections import defaultdict

from constants import log_msg


###############################################################################
# Helper Functions
###############################################################################


def error_recording_hook(exctype, value, tb):
    if exctype == KeyboardInterrupt:
        log_msg("KeyboardInterrupt detected. Exiting...")
    else:
        log_msg(f'Exception: {"".join(traceback.format_exception(exctype, value=value, tb=tb))}', print_out=False)
        sys.__excepthook__(exctype, value, tb)


def parse_arguments():
    # Create the parser
    parser = argparse.ArgumentParser(description='Example script to take command line parameters.')

    # Add arguments
    parser.add_argument('-g', type=str, default=constants.default_gender, help='Speaker Gender: male | female')
    parser.add_argument('-a', type=str, default=constants.default_accent,
                        help=f'Speaker Accent: see {constants.EMBEDDINGS_BASE_PATH}')
    parser.add_argument('-s', type=bool, default=False,
                        help=f'If True, train model using data from specific speaker: see {constants.DATASET_ANALYSIS_PATH}')

    # Parse and return the arguments
    _args = parser.parse_args()

    _accent = _args.a
    _gender = _args.g
    _client_id = ""

    if _args.s:
        # retrieve corresponding metadata
        with (open(constants.DATASET_ANALYSIS_PATH) as client_id_file):
            for _acc, _gen, _client, count in itertools.zip_longest(*[client_id_file] * 4):
                if _acc.strip() == _accent and _gen.strip() == _gender:
                    _client_id = _client.strip()

    return _accent, _gender, _client_id


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

        # TODO: this should be replaced by the corresponding pre-trained embedding
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


def create_speaker_embedding(waveform):
    with torch.no_grad():
        _speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        _speaker_embeddings = torch.nn.functional.normalize(_speaker_embeddings, dim=2)
        _speaker_embeddings = _speaker_embeddings.squeeze().cpu().numpy()
    return _speaker_embeddings


def prepare_dataset(entry):
    """prepare_dataset takes a single entry; tokenize input text; load audio into a log-mel spectrogram; and add
    speaker embeddings"""
    # preserve `client_id` field
    _client_id = entry["client_id"]
    
    # load the entry data; if necessary, this resamples the audio to 16kHz
    audio = entry["audio"]

    # feature extraction and tokenization
    entry = processor(
        text=entry["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    # strip off the batch dimension
    entry["labels"] = entry["labels"][0]

    entry["client_id"] = _client_id

    # use SpeechBrain to obtain x-vector
    # entry["speaker_embeddings"] = create_speaker_embedding(entry["audio"]["array"])

    return entry


def prepare_dataset_batched(entries):
    """prepare_dataset takes a single entry; tokenize input text; load audio into a log-mel spectrogram; and add
    speaker embeddings"""
    # preserve `client_id` field

    chunks = []
    combined = defaultdict(list)

    for i, _ in enumerate(entries['client_id']):
        try:
            _audio = entries['audio'][i]
            _client_id = entries['client_id'][i]

            # feature extraction and tokenization
            entry = processor(
                text=entries['normalized_text'][i],
                audio_target=_audio["array"],
                sampling_rate=_audio["sampling_rate"],
                return_attention_mask=False,
            )
            # print("feature extraction")

            # strip off the batch dimension
            entry["labels"] = entry["labels"][0]
            # print("strip off the batch dimension")

            entry["client_id"] = _client_id
            # print("restore client_id")

            combined['input_ids'].append(entry['input_ids'])
            combined['client_id'].append(entry['client_id'])
            combined['labels'].append(entry['labels'])

            # use SpeechBrain to obtain x-vector
            # entry["speaker_embeddings"] = create_speaker_embedding(entry["audio"]["array"])
        except Exception as e:
            continue

    print(f'\nreturn chunk size: {len(combined["client_id"])}')
    return combined


def is_not_too_short(raw_text, cutoff: int = 10):
    """filters out entry with short text (to prevent T5 from complaining). default cutoff is 10 characters"""
    return len(raw_text) > cutoff


def is_not_too_long(input_ids):
    input_length = len(input_ids)
    return input_length < 200


def has_audio_data(input):
    print(input)
    return input["array"] is not None


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
            row = row[:w - 5] + '...'

        x = max(0, w // 2 - len(row) // 2)  # Ensure x is within the window
        y = max(0, h // 2 - len(directories) // 2 + idx)  # Ensure y is within the window

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
            stdscr.addstr(0, 0, f"You've selected file path: '{dirs[current_row]}'")
            return dirs[current_row]

        print_menu(stdscr, current_row, dirs)


def select_local_mode():
    directories = list(list_directories(constants.MODEL_BASE_PATH))
    if len(directories) == 0:
        log_msg(f"No model under dir {constants.MODEL_BASE_PATH}. Please train and save a model first. Exiting")
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

    # SpeechT5 requires the sampling rate to be 16 kHz.
    log_msg("Start Setting Sampling Rate to 16 kHz...")
    _dataset = _dataset.cast_column("audio", Audio(sampling_rate=16000))
    log_msg("Setting Sampling Rate Successfully")

    return _dataset


def clean_mozilla_dataset(_dataset):
    def normalize_text(entry):
        entry['sentence'] = entry['sentence'].lower()
        return entry

    log_msg(f"Using Mozilla Common Voice. Additional Cleaning Needed")
    log_msg(f"Starting Size of Mozilla Common Voice: {len(_dataset)}")
    table = _dataset.data

    accent_flags = compute.is_in(table['accent'], value_set=pa.array(constants.all_accents, pa.string()))
    table = table.filter(accent_flags)
    print("Size after filtering accent: " + str(len(table)))

    gender_flags = compute.is_in(table['gender'], value_set=pa.array(constants.all_genders, pa.string()))
    table = table.filter(gender_flags)
    print("Size after filtering gender: " + str(len(table)))

    _dataset = datasets.Dataset(table, _dataset.info, _dataset.split)

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
# Speaker Embeddings
###############################################################################
def load_local_speaker_embeddings(_accent, _gender, _device='cpu'):
    embedding_file_path = constants.EMBEDDINGS_BASE_PATH + _accent + '_' + _gender + '.pt'
    try:
        pre_trained_embeddings = torch.load(embedding_file_path)
    except Exception as e:
        log_msg(f"Failed to load local speaker embedding for gender: {_gender} and accent: {_accent}."
                f" Please double check file exists and path is correct. {e}")
        sys.exit()

    pre_trained_embeddings_tensor = torch.tensor(pre_trained_embeddings).unsqueeze(0).to(_device)

    return pre_trained_embeddings_tensor


###############################################################################
# Process Entire Dataset
###############################################################################
def sort_speaker(_dataset):
    # For Mozilla Dataset, the columns (before being mapped on prepare_dataset) are: ['client_id', 'path', 'audio',
    # 'normalized_text', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment']
    print(_dataset)

    # Accents
    up_count = defaultdict(int)
    for up in _dataset["up_votes"]:
        up_count[up] += 1
    ups = list(up_count.keys())
    ups_frequencies = list(up_count.values())
    plt.figure()  # Adjust the figure size as needed
    plt.bar(ups, ups_frequencies)
    plt.title('Frequency of Upvote count')
    plt.xlabel('Upvote')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability
    plt.show()

    # Accents
    accent_count = defaultdict(int)
    for accent_id in _dataset["accent"]:
        accent_count[accent_id] += 1
    accents = list(accent_count.keys())
    accents_frequencies = list(accent_count.values())
    plt.figure()  # Adjust the figure size as needed
    plt.bar(accents, accents_frequencies)
    plt.title('Frequency of Accents')
    plt.xlabel('Accent')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability
    plt.show()

    # Genders
    gender_count = defaultdict(int)
    for gender in _dataset["gender"]:
        gender_count[gender] += 1
    genders = list(gender_count.keys())
    genders_frequencies = list(gender_count.values())
    plt.figure()
    plt.bar(genders, genders_frequencies)
    plt.title('Frequency of Genders')
    plt.xlabel('Gender')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability
    plt.show()

    # Genders
    age_count = defaultdict(int)
    for age in _dataset["age"]:
        age_count[age] += 1
    ages = list(age_count.keys())
    ages_frequencies = list(age_count.values())
    plt.figure()
    plt.bar(ages, ages_frequencies)
    plt.title('Frequency of Ages')
    plt.xlabel('Ages')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotates the x-axis labels for better readability
    plt.show()


def filter_and_prepare_dataset(_dataset) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    log_msg("Start Filtering Short Data")
    _dataset = _dataset.filter(is_not_too_short, input_columns=["normalized_text"])
    log_msg(f"{len(_dataset)} data entries left after filtering short data")

    # log_msg("Start Filtering Audio Data")
    # table = _dataset.data
    # print(str(table['audio'].null_count))
    # _dataset = datasets.Dataset(table, _dataset.info, _dataset.split)
    # # _dataset = _dataset.filter(has_audio_data, input_columns=["audio"])
    # log_msg(f"{len(_dataset)} data entries left after filtering audio data")

    log_msg("Start Preparing Dataset")
    _dataset = _dataset.map(prepare_dataset_batched, batched=True, batch_size=100, remove_columns=_dataset.column_names, num_proc=10)
    log_msg(f"{len(_dataset)} data entries left after preparing dataset in batches")
    print(_dataset)

    log_msg("Start Filtering Long Data")
    _dataset = _dataset.filter(is_not_too_long, input_columns=["input_ids"])
    log_msg(f"{len(_dataset)} data entries left after filtering")

    return _dataset


def filter_dataset_by_client_id(_dataset, _client_id) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    log_msg(f'Start Filtering all data with client_id of: {_client_id}')
    log_msg(f'Starting Size of Dataset: {len(_dataset)}')

    table = _dataset.data
    flags = compute.is_in(table['client_id'], value_set=pa.array([_client_id], pa.string()))

    filtered_table = table.filter(flags)
    filtered_dataset = datasets.Dataset(filtered_table, _dataset.info, _dataset.split)

    log_msg(f"Size after filtering by client_id: {len(filtered_dataset)}")

    return filtered_dataset


def split_dataset(_dataset, _client_id, test_size=constants.dataset_test_size, train_size=constants.dataset_train_size):
    if _client_id != "":
        log_msg(f'client_id provided; train on speaker with id: {_client_id}')
        _dataset = filter_dataset_by_client_id(_dataset, _client_id)
        _dataset = _dataset.train_test_split()
    else:
        log_msg(f'no client_id provided; train on dataset size of: {train_size}')
        _dataset = _dataset.train_test_split(test_size=test_size, train_size=train_size)
    return _dataset


def generate_train_arguments():
    log_msg("Generating Seq2SeqTrainer Arguments")
    return Seq2SeqTrainingArguments(
        output_dir=constants.CHECKPOINT_BASE_PATH,
        per_device_train_batch_size=constants.batch_size,
        gradient_accumulation_steps=constants.gradient_accumulation_steps,
        learning_rate=constants.learning_rate,
        warmup_steps=constants.warm_up_step,
        max_steps=constants.max_steps,
        gradient_checkpointing=False,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=constants.batch_size,
        save_steps=constants.save_steps,
        eval_steps=constants.eval_steps,
        logging_steps=constants.logging_steps,
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


def generate_audio(_spectrogram, _accent, _gender) -> str:
    log_msg(f'Generating output audio file...')
    with torch.no_grad():
        speech = vocoder(_spectrogram)
    speech = speech.cpu()  # move back to CPU

    from IPython.display import Audio
    Audio(speech.numpy(), rate=16000)

    import soundfile as sf
    audio_file_name = f"{constants.AUDIO_OUTPUT_PATH + 'fine_tuned/' + used_model_name + '_' + _accent + '_' + _gender}.wav"
    sf.write(audio_file_name, speech.numpy(), samplerate=16000)

    return audio_file_name


if __name__ == "__main__":
    # Step 0: Parse Commandline Argument
    accent, gender, client_id = parse_arguments()

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

    # Step 2: Add
    speaker_embedding = load_local_speaker_embeddings(accent, gender, device)
    dataset = dataset.add_column("speaker_embeddings", speaker_embedding.tolist() * len(dataset))
    print(dataset)


    # Step 2: Split Dataset. Yes, we load first and then split, this method is more flexible
    divided_dataset = split_dataset(dataset, client_id)

    # Step 3: Train or Load the local model
    used_model_name = None
    if constants.train_model:
        used_model_name = constants.model_name

        pretrained_model.config.use_cache = False
        training_args = generate_train_arguments()
        trainer = generate_trainer(training_args, pretrained_model, divided_dataset, data_collator, tokenizer)
        log_msg(["Using Training Argument: ", json.dumps(training_args.to_dict(), indent=2)])

        log_msg("Start Training...")
        trainer.train()
        log_msg(trainer.state.log_history, include_time=False)

        if constants.save_fine_tuned_model:
            log_msg("Start Saving the model...")
            trainer.save_model(constants.model_path)
        if constants.push_to_hub:
            log_msg("Start Pushing Fine-tuned model to Huggingface")
            trainer.push_to_hub(**constants.huggingface_kwargs)
    else:
        local_model_path = select_local_mode()
        used_model_name = local_model_path.replace(constants.MODEL_BASE_PATH, '').replace('trained_', '')
        pretrained_model = SpeechT5ForTextToSpeech.from_pretrained(pretrained_model_name_or_path=local_model_path,
                                                                   local_files_only=True).to(device)

    # used_model_name = "T5_Vanilla"
    # pretrained_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)

    # Step 4: Generate Audio according to Text and Speaker embeddings
    text = "Jane was working at a diner."
    log_msg(f'Input text: {text}')
    inputs = processor(text=text, return_tensors="pt").to(device)

    # Step 5: Select speaker embeddings (we might generate speaker embeddings on the fly, but there is really no need)
    log_msg(f'Loading Speaker Embedding for {accent} {gender}')
    speaker_embeddings = load_local_speaker_embeddings(accent, gender, device)
    spectrogram = pretrained_model.generate_speech(inputs["input_ids"], speaker_embeddings).to(device)

    # Step 6: Generate Audio
    audio_file_path = generate_audio(spectrogram, accent, gender)

    # Step 7: Add Evaluation Result
    eval_pipeline = pipeline(model=constants.eval_model_name)
    log_msg(f"Evaluating Generated Audio for {audio_file_path}")
    log_msg(eval_pipeline(audio_file_path))

