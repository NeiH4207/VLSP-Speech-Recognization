import torch
import librosa
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


def load_wav_file(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate


def get_mel_spectrogram(path, n_fft=None, win_length=None, hop_length=None, n_mels=None):
    waveform, sample_rate = load_wav_file(path=path)
    # Define transform
    transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=n_mels,
        mel_scale="htk",
    )
    mel_spectrogram = transform(waveform)
    return mel_spectrogram


def get_mfcc_spectrogram(path, n_fft=None, hop_length=None, n_mels=None, n_mfcc=None):
    waveform, sample_rate = load_wav_file(path=path)
    # Define transform
    transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "mel_scale": "htk",
        },
    )
    mfcc = transform(waveform)
    return mfcc


def get_spectral_centroid(path, n_fft=None, win_length=None, hop_length=None):
    waveform, sample_rate = load_wav_file(path=path)
    spectral_centroid = librosa.feature.spectral_centroid(
        y=waveform.numpy(),
        sr=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect"
    )
    return spectral_centroid


def get_spectral_bandwidth(path, n_fft=None, win_length=None, hop_length=None):
    waveform, sample_rate = load_wav_file(path=path)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=waveform.numpy(),
        sr=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect"
    )
    return spectral_bandwidth


if __name__ == "__main__":
    path = "./data/vi/clips/common_voice_vi_21824045.wav"
    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128
    n_mfcc = 128

    print("\nMelSpectrogram")
    print(get_mel_spectrogram(path=path, n_fft=n_fft,
          win_length=win_length, hop_length=hop_length, n_mels=n_mels))
    print("\nMFCC")
    print(get_mfcc_spectrogram(path=path, n_fft=n_fft,
          hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc))
    print("\nSpectral centroid")
    print(get_spectral_centroid(path=path, n_fft=n_fft,
          win_length=win_length, hop_length=hop_length))
    print("\nSpectral bandwidth")
    print(get_spectral_bandwidth(path=path, n_fft=n_fft,
          win_length=win_length, hop_length=hop_length))
