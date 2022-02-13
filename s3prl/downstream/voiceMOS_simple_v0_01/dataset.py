from pathlib import Path
import os

import random
import torch
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate

import pdb

def generate_perturbation():
    perturb_list = ['speed_up', 'speed_down', 'trim', 'pad']
    perturb_type = random.choice(perturb_list)
    
    if perturb_type == 'speed_up':
        ratio = 1.0 + 0.05 * random.random()
        return ['speed', f'{ratio}']
    elif perturb_type == 'speed_down':
        ratio = 1.0 - 0.05 * random.random()
        return ['speed', f'{ratio}']
    elif perturb_type == 'trim':
        ratio = -0.5 * random.random()
        return ['trim', '0', f'{ratio}']
    else:
        ratio = 0.5 * random.random()
        return ['pad', '0', f'{ratio}']

class VoiceMOSDataset(Dataset):
    def __init__(self, mos_list, base_path, valid=False):
        self.base_path = Path(base_path)
        self.mos_list = mos_list
        self.valid = valid

    def __len__(self):
        return len(self.mos_list)

    def __getitem__(self, idx):
        wav_name, mos = self.mos_list.loc[idx]
        wav_path = self.base_path / "wav" / wav_name

        if self.valid:
            wav, _ = apply_effects_file(
                str(wav_path),
                [
                    ["channels", "1"],
                    ["rate", "16000"],
                    ["norm"],
                ],
            )

        else:
            perturb = generate_perturbation()

            wav, _ = apply_effects_file(
                str(wav_path),
                [
                    ["channels", "1"],
                    ["rate", "16000"],
                    perturb,
                    ["norm"],
                ],
            )

        wav = wav.view(-1)
        system_name = wav_name.split("-")[0]

        return wav.numpy(), system_name, mos, wav_name


    def collate_fn(self, samples):
        return zip(*samples)
            
    