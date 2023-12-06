import torch
import os
import time
import sys
import constants
from speechbrain.pretrained import EncoderClassifier
from datasets import Audio, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from datasets import load_dataset, load_from_disk
import numpy
from constants import log_msg
import itertools
import pyarrow as pa
import pyarrow.compute as compute
import datasets

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


def combine_waveform(entry, wc, _accent, _gender, _client_id):
    if entry['client_id'] != _client_id:
        return

    key = _accent + '_' + _gender
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


def filter_by_client_id(_dataset, c_id):
    log_msg(f"Starting Size of Mozilla Common Voice: {len(_dataset)}")
    print("start filtering by client id: " + c_id)
    # _dataset = _dataset.filter(lambda entry: entry["client_id"] == c_id, num_proc=40)

    post_id_test_list = [c_id]
    table = _dataset.data

    flags = compute.is_in(table['client_id'], value_set=pa.array(post_id_test_list, pa.string()))
    filtered_table = table.filter(flags)

    filtered_dataset = datasets.Dataset(filtered_table, _dataset.info, _dataset.split)


    log_msg(f"Size after filtering by client_id: {len(filtered_dataset)}")

    return filtered_dataset


def load_remote_dataset(name="mozilla-foundation/common_voice_1_0",
                        subset=constants.remote_dataset_subset) -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    log_msg(f"Loading Remote Dataset: {name}. Sub-collection: {subset}")
    _dataset = load_dataset(
        name, subset,
        split='other',
        download_mode="reuse_cache_if_exists",
        token=True,
    )

    log_msg("Finish Loading Remote Dataset. Length of dataset: " + str(len(_dataset)))
    # if constants.remote_dataset_name.startswith("mozilla-foundation"):
    #     _dataset = clean_mozilla_dataset(_dataset)

    # return set_sampling_rate(_dataset)
    return _dataset


def load_local_dataset() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    print("Using Locally Processed Dataset. Skip Processing...")

    try:
        _dataset = load_from_disk(constants.data_path)
    except Exception as e:
        print(f"Failed to load local dataset. Please double check file exists and path is correct. {e}")
        sys.exit()

    return _dataset


if __name__ == '__main__':
    # dataset = load_remote_dataset()
    # dataset.save_to_disk(constants.unprocessed_data_path)

    dataset = load_from_disk(constants.unprocessed_data_path)
    dataset = set_sampling_rate(dataset)

    print(dataset)

    waveform_collection = dict()
    client_id_file_name = './src/dataset_analysis/top_client_id.txt'
    with (open(client_id_file_name) as client_id_file):
        for accent, gender, client_id, count in itertools.zip_longest(*[client_id_file] * 4):
            print("Processing audio file for: " + accent.strip() + ", " + gender.strip() + ". Total count: ", count)
            filtered_dataset = filter_by_client_id(dataset, client_id.strip())

            print("Finish filtering using client_id, combining waveform...")
            for e in filtered_dataset:
                combine_waveform(e, waveform_collection, accent.strip(), gender.strip(), client_id.strip())

    print(waveform_collection)
    print(len(waveform_collection.keys()))

    for k, v in waveform_collection.items():
        # first, save the numpy array
        numpy.save('./speaker_embeddings/numpy/' + k, v)
        print("creating speaker embeddings...")

    for k, v in waveform_collection.items():
        speaker_embeddings = create_speaker_embedding(v)

        pt_file_name = './speaker_embeddings/client_id/' + k + '.pt'
        print('saving embedding to: ' + pt_file_name)
        torch.save(speaker_embeddings, pt_file_name)

