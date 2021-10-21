import os
import json_lines
import librosa
import torch
from torch.utils.data import Dataset

# to ignore PySoundFile failed. Trying audioread instead. https://github.com/librosa/librosa/issues/1015
import warnings
warnings.filterwarnings("ignore")


class SpeechDataset(Dataset):
    SAMPLE_RATE = 16000

    def __init__(self, json_file_name: str, root_prefix: str):
        self.root_prefix = root_prefix
        self.audio_filepath = []
        self.text = []

        with open(os.path.join(self.root_prefix, json_file_name), 'rb') as f:
            for item in json_lines.reader(f):
                self.audio_filepath.append(item["audio_filepath"])
                self.text.append(item["text"])

    def __getitem__(self, index: int):
        text = self.text[index]
        path = os.path.join(self.root_prefix, self.audio_filepath[index])
        wav, sr = librosa.load(path, res_type='kaiser_fast')

        if sr != self.SAMPLE_RATE:
            wav = librosa.resample(wav, sr, self.SAMPLE_RATE)

        wav = wav.squeeze()

        instance = {
            'path': path,
            'wav': torch.FloatTensor(wav),
            'text': text
        }

        return instance

    def __len__(self):
        return len(self.audio_filepath)
