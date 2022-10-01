import os

import noisereduce as nr
from scipy.io import wavfile


def reduce_noise(input_path, output_path):
    sample_rate, waveform = wavfile.read(input_path)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(
        y=waveform, 
        sr=sample_rate,
        prop_decrease=1.0,
    )
    wavfile.write(output_path, sample_rate, reduced_noise)
