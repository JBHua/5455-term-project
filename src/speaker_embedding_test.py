import torch
import os
import time
import sys
import constants
from speechbrain.pretrained import EncoderClassifier
from datasets import Audio, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from datasets import load_dataset, load_from_disk


device = "cuda" if torch.cuda.is_available() else "cpu"

def log_msg(message, outf=constants.log_file, include_time=True, print_out=True):
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


speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    run_opts={"device": device},
    savedir=os.path.join("/tmp", "speechbrain/spkrec-xvect-voxceleb")
)


def create_speaker_embedding(waveform):
    with torch.no_grad():
        _speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        _speaker_embeddings = torch.nn.functional.normalize(_speaker_embeddings, dim=2)
        _speaker_embeddings = _speaker_embeddings.squeeze()#.cpu()#.numpy()
    return _speaker_embeddings


def combine_waveform(gender="male", accent=""):

    pass


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
    k = a+g

    if k not in accents:
        accents[k] = 1
    else:
        accents[k] += 1
    
dataset = load_remote_dataset()
print(dataset)
dataset.map(count_accent)

print(accents)

# combine_waveform()

# fp = "./audio_test/combined_16k.mp3"  # change to the correct path to your file accordingly
# signal, sampling_rate = audio2numpy.open_audio(fp)

# print("signal")
# print(signal)

# print("sampling_rate")
# print(sampling_rate)

# embedding = create_speaker_embedding(signal)
# print(embedding)

# torch.save(embedding, '../tensor.pt')
