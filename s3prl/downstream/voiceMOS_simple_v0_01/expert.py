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
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import pdb

from .dataset import VoiceMOSDataset
from .model import Model

warnings.filterwarnings("ignore")


class DownstreamExpert(nn.Module):
    def __init__(self, upstream_dim, downstream_expert, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]
        self.expdir = kwargs["expdir"]

        self.train_dataset = VoiceMOSDataset(
            load_file(self.datarc["voiceMOS_path"], "sets/train_mos_list.txt"),
            self.datarc["voiceMOS_path"],
        )
        self.dev_dataset = VoiceMOSDataset(
            load_file(self.datarc["voiceMOS_path"], "sets/val_mos_list.txt"),
            self.datarc["voiceMOS_path"],
            valid=self.datarc["perturbation"]
        )
        self.test_dataset = VoiceMOSDataset(
            load_file(self.datarc["voiceMOS_path"], "sets/val_mos_list.txt"),
            self.datarc["voiceMOS_path"],
            valid=self.datarc["perturbation"]
        )

        self.system_mos = pd.read_csv(
            Path(
                self.datarc["voiceMOS_path"],
                "system_level_mos.csv",
            ),
            index_col=False
        )


        self.connector = nn.Linear(upstream_dim, self.modelrc["projector_dim"])
        self.model = Model(
            input_dim=self.modelrc["projector_dim"],
            pooling_name=self.modelrc["pooling_name"]
        )
        objective = self.modelrc["objective"]
        self.objective = eval(f"nn.{objective}")()

        self.best_scores = {
            "dev_MSE": np.inf,
            "dev_LCC": -np.inf,
            "dev_SRCC": -np.inf,
        }

    # Interface
    def get_dataloader(self, mode):
        if mode == "train":
            return self._get_train_dataloader(self.train_dataset)
        elif mode == "dev" or mode == "test":
            return self._get_eval_dataloader(self.dev_dataset)



    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset,
            batch_size=self.datarc["train_batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset,
            batch_size=self.datarc["eval_batch_size"],
            shuffle=False,
            num_workers=self.datarc["num_workers"],
            collate_fn=dataset.collate_fn,
        )

    # Interface
    def forward(
        self,
        mode,
        features,
        system_name_list,
        mos_list,
        wav_name_list,
        records,
        **kwargs,
    ):

        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=features[0].device)

        features = pad_sequence(features, batch_first=True)
        features = self.connector(features)

        mos_list = torch.FloatTensor(mos_list).to(features.device)

        uttr_scores = self.model(features, features_len)

        loss = self.objective(uttr_scores, mos_list)

        records["utterance loss"].append(loss.item())

        records["pred_scores"] += uttr_scores.detach().cpu().tolist()
        records["true_scores"] += mos_list.detach().cpu().tolist()

        if mode == "dev" or mode == "test":
            records["wav_names"] += wav_name_list
        
        if len(records["system"]) == 0:
            records["system"].append(defaultdict(list))
        for i in range(len(system_name_list)):
            records["system"][0][system_name_list[i]].append(uttr_scores[i].tolist())

        if mode == "train":
            return loss

        return 0

    # interface
    def log_records(
        self, mode, records, logger, global_step, batch_ids, total_batch_num, **kwargs
    ):
        save_names = []

        # logging loss

        if mode == "train":
            avg_uttr_loss = torch.FloatTensor(records["utterance loss"]).mean().item()

            logger.add_scalar(
                f"voiceMOS/{mode}-utterance loss",
                avg_uttr_loss,
                global_step=global_step,
            )

        # logging Utterance-level MSE, LCC, SRCC

        if mode == "dev":
            # some evaluation-only processing, eg. decoding
            all_pred_scores = records["pred_scores"]
            all_true_scores = records["true_scores"]
            all_pred_scores = np.array(all_pred_scores)
            all_true_scores = np.array(all_true_scores)
            MSE = np.mean((all_true_scores - all_pred_scores) ** 2)
            logger.add_scalar(
                f"voiceMOS/{mode}-Utterance level MSE",
                MSE,
                global_step=global_step,
            )
            pearson_rho, _ = pearsonr(all_true_scores, all_pred_scores)
            logger.add_scalar(
                f"voiceMOS/{mode}-Utterance level LCC",
                pearson_rho,
                global_step=global_step,
            )
            spearman_rho, _ = spearmanr(all_true_scores.T, all_pred_scores.T)
            logger.add_scalar(
                f"voiceMOS/{mode}-Utterance level SRCC",
                spearman_rho,
                global_step=global_step,
            )

            tqdm.write(f"[{mode}] Utterance-level MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] Utterance-level LCC  = {pearson_rho:.4f}")
            tqdm.write(f"[{mode}] Utterance-level SRCC = {spearman_rho:.4f}")

            # select which system mos to use
            system_level_mos = self.system_mos

            # logging System-level MSE, LCC, SRCC
            all_system_pred_scores = []
            all_system_true_scores = []

            for key, values in records["system"][0].items():
                all_system_pred_scores.append(np.mean(values))
                all_system_true_scores.append(system_level_mos[system_level_mos['system_ID']==key]['mean'].values[0])

            all_system_pred_scores = np.array(all_system_pred_scores)
            all_system_true_scores = np.array(all_system_true_scores)

            MSE = np.mean((all_system_true_scores - all_system_pred_scores) ** 2)
            pearson_rho, _ = pearsonr(all_system_true_scores, all_system_pred_scores)
            spearman_rho, _ = spearmanr(all_system_true_scores, all_system_pred_scores)

            tqdm.write(f"[{mode}] System-level MSE  = {MSE:.4f}")
            tqdm.write(f"[{mode}] System-level LCC  = {pearson_rho:.4f}")
            tqdm.write(f"[{mode}] System-level SRCC = {spearman_rho:.4f}")

            logger.add_scalar(
                f"voiceMOS/{mode}-System level MSE",
                MSE,
                global_step=global_step,
            )
            logger.add_scalar(
                f"voiceMOS/{mode}-System level LCC",
                pearson_rho,
                global_step=global_step,
            )
            logger.add_scalar(
                f"voiceMOS/{mode}-System level SRCC",
                spearman_rho,
                global_step=global_step,
            )

        # save model
        if mode == "dev":
            if MSE < self.best_scores["dev_MSE"]:
                self.best_scores["dev_MSE"] = MSE
                save_names.append(f"{mode}-MSE-best.ckpt")
                df = pd.DataFrame(list(zip(records["wav_names"], np.array(records["pred_scores"]))))
                df.to_csv(Path(self.expdir, "MSE-best-answer.txt"), header=None, index=None)
                tqdm.write(f"writing answer.txt")

            if pearson_rho > self.best_scores["dev_LCC"]:
                self.best_scores["dev_LCC"] = pearson_rho
                save_names.append(f"{mode}-LCC-best.ckpt")
                df = pd.DataFrame(list(zip(records["wav_names"], np.array(records["pred_scores"]))))
                df.to_csv(Path(self.expdir, "LCC-best-answer.txt"), header=None, index=None)
                tqdm.write(f"writing answer.txt")

            if spearman_rho > self.best_scores["dev_SRCC"]:
                self.best_scores["dev_SRCC"] = spearman_rho
                save_names.append(f"{mode}-SRCC-best.ckpt")
                df = pd.DataFrame(list(zip(records["wav_names"], np.array(records["pred_scores"]))))
                df.to_csv(Path(self.expdir, "SRCC-best-answer.txt"), header=None, index=None)
                tqdm.write(f"writing answer.txt")

        if mode == "test":
            df = pd.DataFrame(list(zip(records["wav_names"], np.array(records["pred_scores"]))))
            df.to_csv(Path(self.expdir, "answer.txt"), header=None, index=None)

        return save_names


def load_file(base_path, file):
    dataframe = pd.read_csv(Path(base_path, file), header=None)
    return dataframe
