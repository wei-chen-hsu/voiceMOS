from pathlib import Path
import os

import random
import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate

import pdb

class VoiceMOSDataset(Dataset):
    def __init__(self, mos_list, base_path):
        self.base_path = Path(base_path)
        self.mos_list = mos_list

    def __len__(self):
        return len(self.mos_list)

    def __getitem__(self, idx):
        wav_name, mos = self.mos_list.loc[idx]
        wav_path = self.base_path / "wav" / wav_name

        wav, _ = apply_effects_file(
            str(wav_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ],
        )

        wav = wav.view(-1)
        system_name = wav_name.split("-")[0]

        return wav.numpy(), system_name, mos, wav_name


    def collate_fn(self, samples):
        return zip(*samples)
            
    