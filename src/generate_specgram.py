import matplotlib.pyplot as plt
from scipy.io import wavfile
import constants
import socket

file_name = "/Users/yifanhua/PycharmProjects/5455-term-project/audio_eval/us_male/fine_tuned.wav"
# audio_file_name = constants.AUDIO_OUTPUT_PATH + file_name

audio_file_name = file_name

samplingFrequency, signalData = wavfile.read(audio_file_name)
signalData = [i for i in signalData if i != 0]

plt.specgram(signalData,Fs=samplingFrequency)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
