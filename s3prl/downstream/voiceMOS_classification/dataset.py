from pathlib import Path
import os

import random
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torchaudio.sox_effects import apply_effects_file
from itertools import accumulate
from collections import Counter

import pdb

CLASSES = [1,2,3,4,5]

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
    def __init__(self, mos_list, ld_score_list, base_path, idtable = '', valid=False):
        self.base_path = Path(base_path)
        self.mos_list = mos_list
        self.ld_score_list = ld_score_list

        self.class_num = 5
        self.class2index = {CLASSES[i]: i for i in range(len(CLASSES))}

        self.valid = valid
        self._JUDGE = 4

        if not self.valid:
            if Path.is_file(idtable):
                self.idtable = torch.load(idtable)
                for i, judge_i in enumerate(self.ld_score_list[self._JUDGE]):
                    self.ld_score_list[self._JUDGE][i] = self.idtable[judge_i]
            else:
                self.gen_idtable(idtable) 

    def __len__(self):
        if self.valid:
            return len(self.mos_list)
        return len(self.mos_list) + len(self.ld_score_list)

    def __getitem__(self, idx):
        if idx < len(self.mos_list):
            wav_name, mos = self.mos_list.loc[idx]

            wav_ld_score_list = list(self.ld_score_list[self.ld_score_list[1] == wav_name][2])
            wav_ld_score_index_list = [self.class2index[score] for score in wav_ld_score_list]
            wav_ld_score_counter = np.zeros(self.class_num)
            
            for index, value in Counter(wav_ld_score_index_list).items():
                wav_ld_score_counter[index] = value
            
            prob = wav_ld_score_counter / np.sum(wav_ld_score_counter)
            judge_id = 0

        else:
            index = idx - len(self.mos_list)
            system_name, wav_name, opinion_score, _, judge_id = self.ld_score_list.loc[index]
            
            prob = np.zeros(self.class_num)
            prob[self.class2index[opinion_score]] = 1

        wav_path = self.base_path / "wav" / wav_name

        wav, _ = apply_effects_file(
            str(wav_path),
            [
                ["channels", "1"],
                ["rate", "16000"],
                ["norm"],
            ]
        )

        wav = wav.view(-1)
        system_name = wav_name.split("-")[0]

        return wav.numpy(), system_name, prob, wav_name, judge_id


    def collate_fn(self, samples):
        return zip(*samples)

    def gen_idtable(self, idtable_path):
        if idtable_path == '':
            idtable_path = './idtable.pkl'
        self.idtable = {}
        count = 1
        for i, judge_i in enumerate(self.ld_score_list[self._JUDGE]):
            if judge_i not in self.idtable.keys():
                self.idtable[judge_i] = count
                count += 1
                self.ld_score_list[self._JUDGE][i] = self.idtable[judge_i]
            else:
                self.ld_score_list[self._JUDGE][i] = self.idtable[judge_i]
        torch.save(self.idtable, idtable_path)
            
    