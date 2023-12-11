from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, SpeechT5Tokenizer, PreTrainedModel, pipeline
import torch
import constants
from IPython.display import Audio
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

# text = "the brown fox jumps over the lazy dog"
# text = "For the twentieth time that evening the two men shook hands"
# text = "A lemon tea could be great to drink this summer" # US male evaluation
# text = "Are you still using your computer for the research" # Indian female evaluation
# text = "When the judge spoke the death sentence, the defendant showed no emotion" # England Male
# text = "Thousands of years ago, this town was the center of an ancient civilisation" # England Female
# text = "The boy was cuddling with his fluffy teddy bear" #Hongkong male
text = "It seems that the elderly are having difficulties in using the Internet"


# sub_collection = "cmu"
sub_collection = "client_id"
# sub_collection = "collective"

gender = 'male'
# gender = 'female'
accent = 'philippines'

model_name = "Philippines_Male_Dec_9"
# model_name = ""

def load_model():
    if model_name == "":
        return SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    else:
        local_model_path = constants.MODEL_BASE_PATH + model_name
        return SpeechT5ForTextToSpeech.from_pretrained(pretrained_model_name_or_path=local_model_path,
                                                local_files_only=True).to(device)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

inputs = processor(text=text, return_tensors="pt").to(device)

embedding_file_path = constants.EMBEDDINGS_DIR_BASE_PATH + sub_collection + '/' + accent + '_' + gender + '.pt'
pre_trained_embeddings = torch.load(embedding_file_path)
pre_trained_embeddings_tensor = torch.tensor(pre_trained_embeddings).unsqueeze(0).to(device)

loaded_model = load_model()
spectrogram = loaded_model.generate_speech(inputs["input_ids"], pre_trained_embeddings_tensor).to(device)

with torch.no_grad():
    speech = vocoder(spectrogram)
speech = speech.cpu()  # move back to CPU

Audio(speech.numpy(), rate=16000)

if model_name != "":
    audio_file_name = f"{constants.AUDIO_OUTPUT_PATH + 'test/fine_tuned'}.wav"
else:
    audio_file_name = f"{constants.AUDIO_OUTPUT_PATH + 'test/t5'}.wav"

sf.write(audio_file_name, speech.numpy(), samplerate=16000)

eval_pipeline = pipeline(model=constants.eval_model_name)
print(eval_pipeline(audio_file_name))
