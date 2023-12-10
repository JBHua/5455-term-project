import matplotlib.pyplot as plt
from scipy.io import wavfile
import constants
import socket

# file_name = "fine_tuned/Threadripper_Dec_9_18_17_england_female.wav"
file_name = "fine_tuned/trained_Threadripper_Dec_09_22_41_philippines_male.wav"
audio_file_name = constants.AUDIO_OUTPUT_PATH + file_name

samplingFrequency, signalData = wavfile.read(audio_file_name)
plt.specgram(signalData,Fs=samplingFrequency)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()
