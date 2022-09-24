import torch 
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
import os
import pandas as pd

class OutDomainDataset(Dataset):
    def __init__(self, root_dir, infofile,speaker2id, transformation, target_sr , num_samples):
        '''
        root_dir: '/content/vi/clips'
        infofile: train.tsv or test.tsv or dev.tsv
        transformation: MFCC or Mel Spectrogram or...
        target_sr: target sample rate
        num_samples: 
        '''

        self.root_dir = root_dir
        self.infofile = infofile
        self.speaker2id = speaker2id
        self.speakers = pd.read_csv(infofile,sep='\t').loc[:,'client_id']
        self.labels = self.speakers.apply(lambda x: self.speaker2id[x])
        self.filenames =  root_dir + '/' + pd.read_csv(infofile,sep='\t').loc[:,'path']
        assert len(self.speakers) == len(self.filenames)

        self.transformation = transformation
        self.target_sr = target_sr
        self.num_samples = num_samples

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, index):
        audio_path = self.filenames[index].replace('.mp3','.wav')
        label = self.labels[index]
        signal, sr = torchaudio.load(audio_path)
        #signal -> (num_channels, samples) -> (2,16000) -> (1,16000)
        signal = self._resample_if_necessary(signal,sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._return_same_num_samples(signal)
        signal = self.transformation(signal)
        return signal, label

    def _return_same_num_samples(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        elif signal.shape[1] < self.num_samples:
            num_missing_samples = self.num_samples - signal.shape[1]
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal 

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            signal = resampler(signal)
        return signal 

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] == 2:
            signal = torch.mean(signal, dim = 0, keepdim = True)
        return signal