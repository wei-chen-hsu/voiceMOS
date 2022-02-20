import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn as nn
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from tqdm import tqdm
import pdb

from .dataset import VoiceMOSDataset
from .model import Model

warnings.filterwarnings("ignore")

TRUE_SCORE_IDX=0
PRED_SCORE_IDX=1
WAV_NAME_IDX=2


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = kwargs["expdir"]

        self.train_dataset = []
        self.dev_dataset = []
        self.test_dataset = []
        self.system_mos = {}
        self.best_scores = {}

        print(f"[Dataset Information] Using dataset {self.datarc['corpus_names']}........")

        for i, (data_folder, corpus_name) in enumerate(zip(self.datarc['data_folders'], self.datarc['corpus_names'])):
            perturbrc = self.datarc['perturb']

            train_df = load_file(data_folder, self.datarc["train_mos_list_path"])
            train_wav_folder = Path(data_folder) / 'wav'

            self.train_dataset.append(VoiceMOSDataset(mos_list=train_df, 
                                                    wav_folder=train_wav_folder, 
                                                    corpus_name=corpus_name, 
                                                    perturb_mode=perturbrc["mode"], 
                                                    perturb_types=perturbrc["types"], 
                                                    perturb_ratios=perturbrc["ratios"]
                                                    ))

            valid_df = load_file(data_folder, self.datarc["val_mos_list_path"])
            valid_wav_folder = Path(data_folder) / 'wav'

            self.dev_dataset.append(VoiceMOSDataset(mos_list=valid_df, 
                                                    wav_folder=valid_wav_folder, 
                                                    corpus_name=corpus_name, 
                                                    ))

            test_df = load_file(data_folder, self.datarc["test_mos_list_path"])
            test_wav_folder = self.datarc['test_wav_folders'][i] if (len(self.datarc['test_wav_folders']) == len(self.datarc['data_folders'])) else (Path(data_folder) / 'wav')

            self.test_dataset.append(VoiceMOSDataset(mos_list=test_df, 
                                                    wav_folder=test_wav_folder, 
                                                    corpus_name=corpus_name, 
                                                    ))

            self.system_mos[corpus_name] = pd.read_csv(Path(data_folder, "system_level_mos.csv"), index_col=False)
            self.best_scores[corpus_name] = {
                "MSE": 10000,
                "LCC": -1.0,
                "SRCC": -1.0,
            }
        
        self.collate_fn = self.train_dataset[0].collate_fn

        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])
        self.model = Model(
            input_size=self.modelrc["projector_dim"],
            output_size=1,
            pooling_name=self.modelrc["pooling_name"],
            dim=self.modelrc["dim"],
            dropout=self.modelrc["dropout"],
            activation=self.modelrc["activation"]
        )

        print('[Model Information] Printing downstream model information.......')
        print(self.model)

        objective = self.modelrc["objective"]
        self.objective = eval(f"nn.{objective}")()

    # Interface
    def get_dataloader(self, mode):
        if mode == "train":
            return self._get_train_dataloader(ConcatDataset(self.train_dataset))
        elif mode == "dev":
            return self._get_eval_dataloader(ConcatDataset(self.dev_dataset))
        elif mode == "test":
            return self._get_eval_dataloader(ConcatDataset(self.test_dataset))



    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc["train_batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=self.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=self.collate_fn,
        )

    # Interface
    def forward(
        self,
        mode,
        features,
        system_name_list,
        wav_name_list,
        corpus_name_list,
        mos_list,
        records,
        **kwargs,
    ):
        # NOT YET


        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=features[0].device)

        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)

        uttr_scores = self.model(features, features_len)

        mos_list = torch.FloatTensor(mos_list).to(features.device)
        loss = self.objective(uttr_scores, mos_list)

        records["utterance loss"].append(loss.item())

        if mode == "dev" or mode == "test":
            if len(records["all_score"]) == 0:
                for _ in range(3):
                    records["all_score"].append(defaultdict(lambda: defaultdict(list)))

            for corpus_name, system_name, wav_name, uttr_score, mos in zip(corpus_name_list, system_name_list, wav_name_list, uttr_scores.detach().cpu().tolist(), mos_list.detach().cpu().tolist()):
                records["all_score"][PRED_SCORE_IDX][corpus_name][system_name].append(uttr_score)
                records["all_score"][TRUE_SCORE_IDX][corpus_name][system_name].append(mos)
                records["all_score"][WAV_NAME_IDX][corpus_name][system_name].append(wav_name)

        if mode == "train":
            return loss

        return 0

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        save_names = []

        # logging loss

        if mode == "train" or mode == "dev":
            avg_uttr_loss = np.mean(records["utterance loss"])

            logger.add_scalar(
                f"Utterance loss/{mode}",
                avg_uttr_loss,
                global_step=global_step,
            )

        # logging Utterance-level MSE, LCC, SRCC

        if mode == "dev":
            # some evaluation-only processing, eg. decoding
            
            all_system_metric = defaultdict(lambda: defaultdict(float))

            for corpus_name in self.datarc['corpus_names']:
                corpus_pred_score_list = []
                corpus_true_score_list = []
                corpus_wav_name_list = []
                corpus_system_pred_score_list = []
                corpus_system_true_score_list = []

                for system_name in list(records["all_score"][TRUE_SCORE_IDX][corpus_name].keys()):
                    corpus_pred_score_list += records["all_score"][PRED_SCORE_IDX][corpus_name][system_name]
                    corpus_true_score_list += records["all_score"][TRUE_SCORE_IDX][corpus_name][system_name]
                    corpus_wav_name_list += records["all_score"][WAV_NAME_IDX][corpus_name][system_name]
                    corpus_system_pred_score_list.append(np.mean(records["all_score"][PRED_SCORE_IDX][corpus_name][system_name]))
                    corpus_system_true_score_list.append(np.mean(records["all_score"][TRUE_SCORE_IDX][corpus_name][system_name]))
                
                # Calculate utterance level metric 
                corpus_pred_scores = np.array(corpus_pred_score_list)
                corpus_true_scores = np.array(corpus_true_score_list)

                MSE = np.mean((corpus_true_scores - corpus_pred_scores) ** 2)
                LCC, _ = pearsonr(corpus_true_scores, corpus_pred_scores)
                SRCC, _ = spearmanr(corpus_true_scores.T, corpus_pred_scores.T)


                for metric in ['MSE', 'LCC', 'SRCC']:
                    logger.add_scalar(
                        f"Utterance-level/{corpus_name} {mode} {metric}",
                        eval(metric),
                        global_step=global_step,
                    )

                    tqdm.write(f"[{corpus_name}] [{mode}] Utterance-level {metric}  = {eval(metric):.4f}")


                # Calculate system level metric 
                system_level_mos = self.system_mos[corpus_name]

                corpus_system_pred_scores = np.array(corpus_system_pred_score_list)
                corpus_system_true_scores = np.array(corpus_system_true_score_list)

                MSE = np.mean((corpus_system_true_scores - corpus_system_pred_scores) ** 2)
                LCC, _ = pearsonr(corpus_system_true_scores, corpus_system_pred_scores)
                SRCC, _ = spearmanr(corpus_system_true_scores, corpus_system_pred_scores)

                for metric in ['MSE', 'LCC', 'SRCC']:
                    all_system_metric[corpus_name][metric] = eval(metric)

                    logger.add_scalar(
                        f"System-level/{corpus_name} {mode} {metric}",
                        eval(metric),
                        global_step=global_step,
                    )

                    tqdm.write(f"[{corpus_name}] [{mode}] System-level {metric}  = {eval(metric):.4f}")

        if mode == "dev" or mode == "test":
            all_pred_score_list = []
            all_wav_name_list = []

            for corpus_name in self.datarc['corpus_names']:
                for system_name in list(records["all_score"][PRED_SCORE_IDX][corpus_name].keys()):
                    all_pred_score_list += records["all_score"][PRED_SCORE_IDX][corpus_name][system_name]
                    all_wav_name_list += records["all_score"][WAV_NAME_IDX][corpus_name][system_name]

        if mode == "dev":
            for corpus_name in self.datarc['corpus_names']:
                for metric, operator in zip(['MSE', 'LCC', 'SRCC'], ["<", ">", ">"]):
                    if eval(f"{all_system_metric[corpus_name][metric]} {operator} {self.best_scores[corpus_name][metric]}"):
                        self.best_scores[corpus_name][metric] = all_system_metric[corpus_name][metric]
                        save_names.append(f"{mode}-{corpus_name}-{metric}-best.ckpt")
                        df = pd.DataFrame(list(zip(all_wav_name_list, all_pred_score_list)))
                        df.to_csv(Path(self.expdir, f"{mode}-{corpus_name}-{metric}-best-answer.txt"), header=None, index=None)

        if mode == "test":
            df = pd.DataFrame(list(zip(all_wav_name_list, all_pred_score_list)))
            df.to_csv(Path(self.expdir, f"{mode}-steps-{global_step}-answer.txt"), header=None, index=None)

        return save_names


def load_file(base_path, file):
    dataframe = pd.read_csv(Path(base_path, file), header=None)
    return dataframe
