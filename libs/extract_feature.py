import librosa
import torchaudio.transforms as T
from src.utils import *
from configs import config

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
    # Testing
    path = config.FeatureExtraction['sample-path']
    n_fft = config.FeatureExtraction['n-fft']
    win_length = config.FeatureExtraction['win-length']
    hop_length = config.FeatureExtraction['hop-length']
    n_mels = config.FeatureExtraction['n-mel']
    n_mfcc = config.FeatureExtraction['n-mfcc']

    from matplotlib import pyplot as plt
    import matplotlib
    import librosa.display
    matplotlib.style.use('ggplot')
    
    mel_spectrogram = get_mel_spectrogram(path=path, n_fft=n_fft,
          win_length=win_length, hop_length=hop_length, n_mels=n_mels)
    
    mfcc_spectrogram = get_mfcc_spectrogram(path=path, n_fft=n_fft,
          hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
    
    spectral_centroid = get_spectral_centroid(path=path, n_fft=n_fft,
          win_length=win_length, hop_length=hop_length)
    
    spectral_bandwidth = get_spectral_bandwidth(path=path, n_fft=n_fft,
          win_length=win_length, hop_length=hop_length)
    
    librosa.display.specshow(mel_spectrogram[0].numpy())
    plt.title('Mel Spectrogram')
    plt.colorbar(format="%+2.f")
    plt.show()
    
    librosa.display.specshow(mfcc_spectrogram[0].numpy())
    plt.title('MFCC')
    plt.colorbar(format="%+2.f")
    plt.show()
    
    figure, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].set_title("Spectral centroid")
    axes[0].plot(spectral_centroid[0][0])
    axes[1].plot(spectral_bandwidth[0][0])
    axes[1].set_title("Spectral bandwidth")
    plt.show()
