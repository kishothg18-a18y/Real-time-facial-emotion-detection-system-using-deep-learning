import sounddevice as sd
from scipy.io.wavfile import write
import os

def record_voice(filename="voice_input.wav", duration=4, fs=22050):
    """
    Records voice from microphone and saves it as a WAV file.
    :param filename: Output audio file name
    :param duration: Recording duration in seconds
    :param fs: Sampling frequency
    """
    print("Recording voice... Please speak.")
    
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()  # Wait until recording is finished
    
    write(filename, fs, recording)
    
    if os.path.exists(filename):
        print(f"Voice recorded and saved as {filename}")
    else:
        raise Exception("Voice recording failed")
