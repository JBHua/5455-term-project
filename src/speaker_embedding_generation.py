import torch
import os
import time
import sys
import constants
from speechbrain.pretrained import EncoderClassifier
from datasets import Audio, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from datasets import load_dataset, load_from_disk
import numpy
from src.constants import log_msg
from src.constants import all_genders

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # using cuda will result in VRAM exhaustion, unless u have a GPU with 48GB of VRAM


speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    run_opts={"device": device},
    savedir=os.path.join("/tmp", "speechbrain/spkrec-xvect-voxceleb")
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        _speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        _speaker_embeddings = torch.nn.functional.normalize(_speaker_embeddings, dim=2)
        _speaker_embeddings = _speaker_embeddings.squeeze().cpu().numpy()
    return _speaker_embeddings


def combine_waveform(entry, wc):
    if entry['gender'] not in constants.all_genders:
        pass

    if entry['accent'] not in constants.all_accents:
        pass

    key = entry['accent'] + '_' + entry['gender']
    if key not in wc:
        wc[key] = entry['audio']['array']
    else:
        wc[key] = numpy.concatenate((wc[key], entry['audio']['array']))


def set_sampling_rate(_dataset):
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
        download_mode="reuse_cache_if_exists"
    )

    log_msg("Finish Loading Remote Dataset. Length of dataset: " + str(len(_dataset)))
    if constants.remote_dataset_name.startswith("mozilla-foundation"):
        _dataset = clean_mozilla_dataset(_dataset)

    return set_sampling_rate(_dataset)


def load_local_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    print("Using Locally Processed Dataset. Skip Processing...")

    try:
        _dataset = load_from_disk(constants.data_path)
    except Exception as e:
        print(f"Failed to load local dataset. Please double check file exists and path is correct. {e}")
        sys.exit()

    return _dataset


accents = dict()


def count_accent(entry):
    a = entry["accent"]
    g = entry["gender"]
    k = a + " " + g

    if k not in accents:
        accents[k] = 1
    else:
        accents[k] += 1


def set_sampling_rate(_dataset):
    from datasets import Audio

    log_msg("Start Setting Sampling Rate to 16 kHz...")
    # SpeechT5 requires the sampling rate to be 16 kHz.
    _dataset = _dataset.cast_column("audio", Audio(sampling_rate=16000))
    log_msg("Setting Sampling Rate Successfully")

    return _dataset


# dataset = load_remote_dataset()
# dataset.save_to_disk(constants.unprocessed_data_path)
dataset = load_from_disk(constants.unprocessed_data_path)
print(dataset)

dataset = set_sampling_rate(dataset)
print(dataset)

waveform_collection = dict()

i = 0
for e in dataset:
    combine_waveform(e, waveform_collection)
    i += 1
    if i % 500 == 0:
        print(f'{i*100/8637}%')

print(waveform_collection)
print(len(waveform_collection.keys()))

for k, v in waveform_collection.items():
    embedding = create_speaker_embedding(v)
    filename = './speaker_embeddings/' + k + '.pt'
    print('saving embedding to: ' + filename)
    torch.save(embedding, filename)
