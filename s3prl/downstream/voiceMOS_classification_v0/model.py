import torch
import torch.nn as nn

from s3prl.downstream.model import AttentivePooling, MeanPooling

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)


    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep


class Model(nn.Module):
    def __init__(self, input_dim, pooling_name, class_num, num_judges=10000, **kwargs):
        super(Model, self).__init__()
        self.class_num = class_num
        self.linear = nn.Linear(input_dim, class_num)
        self.pooling = eval(pooling_name)(input_dim=input_dim, activation='ReLU')
        self.judge_embbeding = nn.Embedding(num_embeddings = num_judges, embedding_dim=input_dim)

    def forward(self, features, features_len, judge_ids):
        time = features.shape[1]
        judge_features = self.judge_embbeding(judge_ids)
        judge_features = torch.stack([judge_features for i in range(time)], dim = 1)
        features = features + judge_features

        features, _ = self.pooling(features, features_len)
        logits = self.linear(features)


        return logits.squeeze(-1)
