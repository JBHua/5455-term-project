from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SpeechT5Tokenizer, PreTrainedModel, pipeline
import torch
import constants
from IPython.display import Audio
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

use_client_id_version = True
# gender = 'male'
gender = 'female'
accent = 'us'

t5_vanilla_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

text = "The freedom to be a kid without being influenced by the internet"

inputs = processor(text=text, return_tensors="pt").to(device)

if use_client_id_version:
    embedding_file_path = constants.EMBEDDINGS_BASE_PATH + 'client_id/' + accent + '_' + gender + '.pt'
else:
    embedding_file_path = constants.EMBEDDINGS_BASE_PATH + 'collective/' + accent + '_' + gender + '.pt'

pre_trained_embeddings = torch.load(embedding_file_path)
pre_trained_embeddings_tensor = torch.tensor(pre_trained_embeddings).unsqueeze(0).to(device)

spectrogram = t5_vanilla_model.generate_speech(inputs["input_ids"], pre_trained_embeddings_tensor).to(device)

with torch.no_grad():
    speech = vocoder(spectrogram)
speech = speech.cpu()  # move back to CPU

Audio(speech.numpy(), rate=16000)

if use_client_id_version:
    audio_file_name = f"{constants.AUDIO_OUTPUT_PATH + 't5_vanilla/' + 'client_id_' + accent + '_' + gender}.wav"
else:
    audio_file_name = f"{constants.AUDIO_OUTPUT_PATH + 't5_vanilla/' + accent + '_' + gender}.wav"


sf.write(audio_file_name, speech.numpy(), samplerate=16000)

eval_pipeline = pipeline(model=constants.eval_model_name)
print(eval_pipeline(audio_file_name))
