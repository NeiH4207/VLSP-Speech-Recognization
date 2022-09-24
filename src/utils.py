import os
import torchaudio

def load_wav_file(path):
    waveform, sample_rate = torchaudio.load(path)
    return waveform, sample_rate

def gather_file(path, ext='.wav'):
    all_files = os.listdir(path)
    return [_file for _file in all_files if _file.endswith(ext)]

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)
