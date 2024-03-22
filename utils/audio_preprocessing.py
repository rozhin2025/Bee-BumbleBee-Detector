import librosa
import numpy as np
from scipy import signal

def segmentation(audio, sr=22050, frame_length=None, hop_length=None):
    '''
    Parameters:
    audio: audio data
    sr: sampling rate
    frame_length: length of each frame
    hop_length: number of samples between successive frames
    
    Returns:
    audio segments
    '''
    if frame_length is None: frame_length = sr
    if hop_length is None: hop_length = sr//4
    return librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length, writeable=True).T


def normalize(audio_segment, norm=np.inf):
    '''
    Parameters:
    audio_segment: audio data
    norm: order of norm
    
    Returns:
    normalized audio segment
    '''
    return librosa.util.normalize(audio_segment, norm=norm, axis=-1)


# loads audio file using librosa library with 
# - sampling rate of 22050 (we want to remove frequencies above 11025 Hz according to antialiasing theorem and bee frequency range[100hz, 10khz] in litretures)
# - mono channel (single channel)
# - 16 bit floating point precision by default
# - normalizes the audio file to [-1,1] range by default
# - soxr’s high-quality mode (‘HQ’) resampling by default
# returns audio data and sampling rate
def preprocessing(file_path, sr=22050):
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    sos = signal.butter(8, 150, btype='high', fs=sr, output='sos')
    audio = signal.sosfiltfilt(sos, audio)
    return audio, sr


def complete_preprocessing(file_path, sr=22050, frame_length=None, hop_length=None, norm=np.inf):
    audio, sr = preprocessing(file_path, sr)
    audio_segments = segmentation(audio, sr, frame_length, hop_length)
    audio_segments = normalize(audio_segments, norm)
    return audio_segments, sr

if __name__ == "__main__":
    pass
