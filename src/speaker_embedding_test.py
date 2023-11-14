import torch
import os
from speechbrain.pretrained import EncoderClassifier
from datasets import Audio
from pydub import AudioSegment
import audio2numpy


device = "cuda" if torch.cuda.is_available() else "cpu"

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


fp = "./audio_test/combined_16k.mp3"  # change to the correct path to your file accordingly
signal, sampling_rate = audio2numpy.open_audio(fp)

print("signal")
print(signal)

print("sampling_rate")
print(sampling_rate)

embedding = create_speaker_embedding(signal)
print(embedding)

torch.save(embedding, '../tensor.pt')
