from pathlib import Path
import os

import random
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate

import pdb

def generate_apply_effect_file(length, valid=False):
    perturb_list = ['speed_up', 'speed_down', 'trim', 'pad']
    perturb_type = random.choice(perturb_list)
    rng = random.choice(np.linspace(0,1,101))
    
    apply_effect_file_list = []
    for i in range(length):
        apply_effect_file_list.append([
                    ["channels", "1"],
                    ["rate", "16000"],
                    ["norm"],
                ])

    if valid == False:
        all_perturbation_list = []
        for i in range(length):
            ratio = 1.0 + 0.05 * rng
            all_perturbation_list.append(['speed', f'{ratio}'])
        for i in range(length):
            ratio = 1.0 - 0.05 * rng
            all_perturbation_list.append(['speed', f'{ratio}'])
        for i in range(length):
            ratio = -0.5 * rng
            all_perturbation_list.append(['trim', '0', f'{ratio}'])
        for i in range(length):
            ratio = 0.5 * rng
            all_perturbation_list.append(['pad', '0', f'{ratio}'])

        for perturb in all_perturbation_list:
            apply_effect_file_list.append([
                        ["channels", "1"],
                        ["rate", "16000"],
                        perturb,
                        ["norm"],
                    ])

    return apply_effect_file_list

class VoiceMOSDataset(Dataset):
    def __init__(self, mos_list, base_path, valid=False):
        self.base_path = Path(base_path)
        self.mos_list = mos_list
        self.valid = valid
        self.apply_effect_file_list = generate_apply_effect_file(length=len(self.mos_list), valid=self.valid)

        print(len(self.apply_effect_file_list))

    def __len__(self):
        return len(self.apply_effect_file_list)

    def __getitem__(self, idx):
        wav_name, mos = self.mos_list.loc[(idx % len(self.mos_list))]
        wav_path = self.base_path / "wav" / wav_name

        wav, _ = apply_effects_file(
            str(wav_path),
            self.apply_effect_file_list[idx]
        )

        wav = wav.view(-1)
        system_name = wav_name.split("-")[0]

        return wav.numpy(), system_name, mos, wav_name


    def collate_fn(self, samples):
        return zip(*samples)
            
    