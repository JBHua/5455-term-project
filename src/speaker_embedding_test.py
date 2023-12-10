from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SpeechT5Tokenizer, PreTrainedModel, pipeline
import torch
import constants
from IPython.display import Audio
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

# sub_collection = "cmu"
sub_collection = "client_id"
# sub_collection = "collective"

# gender = 'male'
gender = 'female'
accent = 'us'

t5_vanilla_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

text = "the brown fox jumps over the lazy dog"

inputs = processor(text=text, return_tensors="pt").to(device)

embedding_file_path = constants.EMBEDDINGS_DIR_BASE_PATH + sub_collection + '/' + accent + '_' + gender + '.pt'
pre_trained_embeddings = torch.load(embedding_file_path)
pre_trained_embeddings_tensor = torch.tensor(pre_trained_embeddings).unsqueeze(0).to(device)

spectrogram = t5_vanilla_model.generate_speech(inputs["input_ids"], pre_trained_embeddings_tensor).to(device)

with torch.no_grad():
    speech = vocoder(spectrogram)
speech = speech.cpu()  # move back to CPU

Audio(speech.numpy(), rate=16000)

audio_file_name = f"{constants.AUDIO_OUTPUT_PATH + 't5_vanilla/' + sub_collection + '_' + accent + '_' + gender}.wav"

sf.write(audio_file_name, speech.numpy(), samplerate=16000)

eval_pipeline = pipeline(model=constants.eval_model_name)
print(eval_pipeline(audio_file_name))
