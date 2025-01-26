from types import SimpleNamespace

import os
import pickle as pk
from functools import lru_cache

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..featurizer.protein import FOLDSEEK_MISSING_IDX
from ..utils import get_logger

logg = get_logger()

#################################
# Latent Space Distance Metrics #
#################################


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


DISTANCE_METRICS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
}

ACTIVATIONS = {"ReLU": nn.ReLU, "GELU": nn.GELU, "ELU": nn.ELU, "Sigmoid": nn.Sigmoid}

#######################
# Model Architectures #
#######################


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`
    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]), requires_grad=False)
        self.k.requiresGrad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise
        :param x: :math:`(N \\times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \\times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(
            1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1
        ).squeeze()
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.
        :meta private:
        """
        self.k.data.clamp_(min=0)


#######################
# Model Architectures #
#######################


class SimpleCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        # dropout_rate=0.2, 
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]
        # self.dropout_rate=dropout_rate

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), self.latent_activation()
            # ,nn.Dropout(self.dropout_rate)
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), self.latent_activation()
            # ,nn.Dropout(self.dropout_rate)
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        # drug = drug.to(torch.float32)
        # target = target.to(torch.float32)
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        # if drug.dtype != torch.float32:
        #     drug = drug.float()
        # if target.dtype != torch.float32:
        #     target = target.float()

        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()



SimpleCoembeddingNoSigmoid = SimpleCoembedding


# class AttentionInteraction(nn.Module):
#     def __init__(self, drug_shape=2048, target1_shape=1024, target2_shape=512, target3_shape=256, latent_dimension=1024):
#         super().__init__()
#         self.drug_shape = drug_shape
#         self.target1_shape = target1_shape
#         self.target2_shape = target2_shape
#         self.target3_shape = target3_shape
#         self.latent_dimension = latent_dimension

#         # 投影层
#         self.drug_projector = nn.Linear(self.drug_shape, self.latent_dimension)
        
#         # 单独初始化每个靶标的投影层
#         self.target1_projector = nn.Linear(self.target1_shape, self.latent_dimension)
#         self.target2_projector = nn.Linear(self.target2_shape, self.latent_dimension)
#         self.target3_projector = nn.Linear(self.target3_shape, self.latent_dimension)

#         # 注意力层
#         self.attention_layer = nn.Linear(self.latent_dimension, 1)

#     def forward(self, drug, target1, target2, target3):
#         # drug: (batch_size, drug_shape)
#         # target1: (batch_size, target1_shape)
#         # target2: (batch_size, target2_shape)
#         # target3: (batch_size, target3_shape)

#         # 投影药物
#         drug_projection = self.drug_projector(drug)  # (batch_size, latent_dimension)

#         # 投影靶标
#         target1_projection = self.target1_projector(target1)  # (batch_size, latent_dimension)
#         target2_projection = self.target2_projector(target2)  # (batch_size, latent_dimension)
#         target3_projection = self.target3_projector(target3)  # (batch_size, latent_dimension)

#         # 将靶标投影堆叠成一个张量
#         target_projections = torch.stack([target1_projection, target2_projection, target3_projection], dim=1)  # (batch_size, num_targets, latent_dimension)

#         # 计算注意力权重
#         attention_scores = self.attention_layer(target_projections)  # (batch_size, num_targets, 1)
#         attention_weights = torch.softmax(attention_scores, dim=1)  # (batch_size, num_targets, 1)

#         # 计算加权和
#         weighted_target_sum = (attention_weights * target_projections).sum(dim=1)  # (batch_size, latent_dimension)

#         # 计算最终输出
#         final_output = torch.bmm(drug_projection.view(-1, 1, self.latent_dimension), weighted_target_sum.view(-1, self.latent_dimension, 1)).squeeze()

#         return final_output

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation, dropout_rate=0.5):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)  # 添加 Batch Normalization
        self.activation = activation()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)  # 添加 Batch Normalization

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.bn1(out)  # Batch Normalization
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.bn2(out)  # Batch Normalization
        out += residual
        return self.activation(out)

# class ResidualBlockshort(nn.Module):
#     def __init__(self, input_dim, output_dim, activation, dropout_rate=0.5):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, output_dim)
#         self.bn1 = nn.BatchNorm1d(output_dim)
#         self.activation = activation()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.linear2 = nn.Linear(output_dim, output_dim)
#         self.bn2 = nn.BatchNorm1d(output_dim)
#         self.shortcut = nn.Sequential(
#             nn.Linear(input_dim, output_dim),
#             nn.BatchNorm1d(output_dim)
#         )

#     def forward(self, x):
#         residual = self.shortcut(x)
#         out = self.linear1(x)
#         out = self.bn1(out)
#         out = self.activation(out)
#         out = self.dropout(out)
#         out = self.linear2(out)
#         out = self.bn2(out)
#         out = self.activation(out)
#         out += residual
#         return out


class SimpleCoembeddingResnet(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        num_residual_blocks=2,  # 添加残差块的数量
        dropout_rate=0.5,  # Dropout 比例
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        # 药物投影
        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension),
            self.latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        # 添加药物的残差块
        self.drug_residual_blocks = nn.Sequential(
            *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
        )

        # 靶标投影
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension),
            self.latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        # 添加靶标的残差块
        self.target_residual_blocks = nn.Sequential(
            *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
        )

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        drug_projection = self.drug_residual_blocks(drug_projection)

        target_projection = self.target_projector(target)
        target_projection = self.target_residual_blocks(target_projection)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        drug_projection = self.drug_residual_blocks(drug_projection)

        target_projection = self.target_projector(target)
        target_projection = self.target_residual_blocks(target_projection)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


import torch
import torch.nn as nn

class SimpleCoembeddingResnet3(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        target_shape_1=1024,
        target_shape_2=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        num_residual_blocks=2,  # 添加残差块的数量
        dropout_rate=0.5,  # Dropout 比例
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.target_shape_1 = target_shape_1
        self.target_shape_2 = target_shape_2
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        # 药物投影
        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension),
            self.latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        # 添加药物的残差块
        self.drug_residual_blocks = nn.Sequential(
            *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
        )

        # 靶标投影
        self.target_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.target_shape, latent_dimension),
                self.latent_activation()
            ),
            nn.Sequential(
                nn.Linear(self.target_shape_1, latent_dimension),
                self.latent_activation()
            ),
            nn.Sequential(
                nn.Linear(self.target_shape_2, latent_dimension),
                self.latent_activation()
            )
        ])
        for projector in self.target_projectors:
            nn.init.xavier_normal_(projector[0].weight)

        # 添加靶标的残差块
        self.target_residual_blocks = nn.ModuleList([
            nn.Sequential(
                *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
            ) for _ in range(3)
        ])

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target, target1, target2):
        if self.do_classify:
            return self.classify(drug, target, target1, target2)
        else:
            return self.regress(drug, target, target1, target2)

    def regress(self, drug, target, target1, target2):
        drug_projection = self.drug_projector(drug)
        drug_projection = self.drug_residual_blocks(drug_projection)

        target_projections = []
        for i, (target, projector, residual_blocks) in enumerate(zip([target, target1, target2], self.target_projectors, self.target_residual_blocks)):
            target_projection = projector(target)
            target_projection = residual_blocks(target_projection)
            target_projections.append(target_projection)

        inner_prods = []
        for target_projection in target_projections:
            inner_prod = torch.bmm(
                drug_projection.view(-1, 1, self.latent_dimension),
                target_projection.view(-1, self.latent_dimension, 1),
            ).squeeze()
            inner_prods.append(inner_prod)

        return inner_prods

    def classify(self, drug, target, target1, target2):
        drug_projection = self.drug_projector(drug)
        drug_projection = self.drug_residual_blocks(drug_projection)

        target_projections = []
        for i, (target, projector, residual_blocks) in enumerate(zip([target, target1, target2], self.target_projectors, self.target_residual_blocks)):
            target_projection = projector(target)
            target_projection = residual_blocks(target_projection)
            target_projections.append(target_projection)

        distances = []
        for target_projection in target_projections:
            distance = self.activator(drug_projection, target_projection)
            distances.append(distance)

        return distances
    
# class SimpleCoembeddingResnet3_no(nn.Module):
#     def __init__(
#         self,
#         drug_shape=2048,
#         target_shape=1024,
#         target_shape_1=1024,
#         target_shape_2=1024,
#         latent_dimension=1024,
#         latent_activation="ReLU",
#         latent_distance="Cosine",
#         classify=True,
#         num_residual_blocks=2,  # 添加残差块的数量
#         dropout_rate=0.5,  # Dropout 比例
#     ):
#         super().__init__()
#         self.drug_shape = drug_shape
#         self.target_shape = target_shape
#         self.target_shape_1 = target_shape_1
#         self.target_shape_2 = target_shape_2
#         self.latent_dimension = latent_dimension
#         self.do_classify = classify
#         self.latent_activation = ACTIVATIONS[latent_activation]

#         # 药物投影
#         self.drug_projector = nn.Sequential(
#             nn.Linear(self.drug_shape, latent_dimension),
#             self.latent_activation()
#         )
#         nn.init.xavier_normal_(self.drug_projector[0].weight)

#         # 添加药物的残差块
#         self.drug_residual_blocks = nn.Sequential(
#             *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
#         )

#         # 靶标投影
#         self.target_projectors = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.target_shape, latent_dimension),
#                 self.latent_activation()
#             ),
#             nn.Sequential(
#                 nn.Linear(self.target_shape_1, latent_dimension),
#                 self.latent_activation()
#             ),
#             nn.Sequential(
#                 nn.Linear(self.target_shape_2, latent_dimension),
#                 self.latent_activation()
#             )
#         ])
#         for projector in self.target_projectors:
#             nn.init.xavier_normal_(projector[0].weight)

#         # 添加靶标的残差块（仅针对前个靶标）
#         self.target_residual_blocks = nn.ModuleList([
#             nn.Sequential(
#                 *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
#             ) for _ in range(1)  # 仅为前个靶标添加残差块
#         ])

#         if self.do_classify:
#             self.distance_metric = latent_distance
#             self.activator = DISTANCE_METRICS[self.distance_metric]()

#     def forward(self, drug, target, target1, target2):
#         if self.do_classify:
#             return self.classify(drug, target, target1, target2)
#         else:
#             return self.regress(drug, target, target1, target2)

#     def regress(self, drug, target, target1, target2):
#         drug_projection = self.drug_projector(drug)
#         drug_projection = self.drug_residual_blocks(drug_projection)

#         target_projections = []
#         for i, (target, projector) in enumerate(zip([target, target1, target2], self.target_projectors)):
#             target_projection = projector(target)
#             if i < 1:  # 仅对前两个靶标应用残差块
#                 target_projection = self.target_residual_blocks[i](target_projection)
#             target_projections.append(target_projection)

#         inner_prods = []
#         for target_projection in target_projections:
#             inner_prod = torch.bmm(
#                 drug_projection.view(-1, 1, self.latent_dimension),
#                 target_projection.view(-1, self.latent_dimension, 1),
#             ).squeeze()
#             inner_prods.append(inner_prod)

#         return inner_prods

#     def classify(self, drug, target, target1, target2):
#         drug_projection = self.drug_projector(drug)
#         drug_projection = self.drug_residual_blocks(drug_projection)

#         target_projections = []
#         for i, (target, projector) in enumerate(zip([target, target1, target2], self.target_projectors)):
#             target_projection = projector(target)
#             if i < 1:  # 仅对前两个靶标应用残差块
#                 target_projection = self.target_residual_blocks[i](target_projection)
#             target_projections.append(target_projection)

#         distances = []
#         for target_projection in target_projections:
#             distance = self.activator(drug_projection, target_projection)
#             distances.append(distance)

#         return distances

class SimpleCoembeddingResnet3_no(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        target_shape_1=1024,
        target_shape_2=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        num_residual_blocks=2,  # 添加残差块的数量
        dropout_rate=0.5,  # Dropout 比例
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.target_shape_1 = target_shape_1
        self.target_shape_2 = target_shape_2
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        # 为每个目标创建独立的药物投影层
        self.drug_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.drug_shape, latent_dimension),
                self.latent_activation()
            ) for _ in range(3)  # 假设有三个目标
        ])
        
        for projector in self.drug_projectors:
            nn.init.xavier_normal_(projector[0].weight)

        # 添加第一个药物的残差块
        self.drug_residual_block = nn.Sequential(
            *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
        )

        # 靶标投影
        self.target_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.target_shape, latent_dimension),
                self.latent_activation()
            ),
            nn.Sequential(
                nn.Linear(self.target_shape_1, latent_dimension),
                self.latent_activation()
            ),
            nn.Sequential(
                nn.Linear(self.target_shape_2, latent_dimension),
                self.latent_activation()
            )
        ])
        for projector in self.target_projectors:
            nn.init.xavier_normal_(projector[0].weight)

        # 添加第一个靶标的残差块
        self.target_residual_block = nn.Sequential(
            *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
        )

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target, target1, target2):
        if self.do_classify:
            return self.classify(drug, target, target1, target2)
        else:
            return self.regress(drug, target, target1, target2)

    def regress(self, drug, target, target1, target2):
        inner_prods = []

        # 处理第一个药物和第一个目标
        drug_projection = self.drug_projectors[0](drug)
        drug_projection = self.drug_residual_block(drug_projection)

        target_projection = self.target_projectors[0](target)
        target_projection = self.target_residual_block(target_projection)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        inner_prods.append(inner_prod)

        # 处理第二个和第三个目标
        for i, (target, target_projector) in enumerate(zip([target1, target2], self.target_projectors[1:])):
            drug_projection = self.drug_projectors[i + 1](drug)  # 只需处理药物投影而不使用残差块
            target_projection = target_projector(target)

            inner_prod = torch.bmm(
                drug_projection.view(-1, 1, self.latent_dimension),
                target_projection.view(-1, self.latent_dimension, 1),
            ).squeeze()
            inner_prods.append(inner_prod)

        return inner_prods

    def classify(self, drug, target, target1, target2):
        distances = []

        # 处理第一个药物和第一个目标
        drug_projection = self.drug_projectors[0](drug)
        drug_projection = self.drug_residual_block(drug_projection)

        target_projection = self.target_projectors[0](target)
        target_projection = self.target_residual_block(target_projection)

        distance = self.activator(drug_projection, target_projection)
        distances.append(distance)

        # 处理第二个和第三个目标
        for i, (target, target_projector) in enumerate(zip([target1, target2], self.target_projectors[1:])):
            drug_projection = self.drug_projectors[i + 1](drug)  # 只需处理药物投影而不使用残差块
            target_projection = target_projector(target)

            distance = self.activator(drug_projection, target_projection)
            distances.append(distance)

        return distances



import torch
import torch.nn as nn

class SimpleCoembeddingResnet3_no_d(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        target_shape_1=1024,
        target_shape_2=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        num_residual_blocks=2,  # 添加残差块的数量
        dropout_rate=0.5,  # Dropout 比例
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.target_shape_1 = target_shape_1
        self.target_shape_2 = target_shape_2
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        
        # 获取激活函数
        self.latent_activation = ACTIVATIONS[latent_activation]
        
        # 创建三个独立的药物和目标网络
        self.networks = nn.ModuleList([
            self.create_network(drug_shape, target_shape, latent_dimension, num_residual_blocks, dropout_rate),  # 第一个对
            self.create_simple_network(drug_shape, target_shape_1, latent_dimension),  # 第二个对
            self.create_simple_network(drug_shape, target_shape_2, latent_dimension)   # 第三个对
        ])

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def create_network(self, drug_shape, target_shape, latent_dimension, num_residual_blocks, dropout_rate):
        # 创建一个子网络，包括药物投影、靶标投影和残差块
        return nn.ModuleDict({
            'drug_projector': nn.Sequential(
                nn.Linear(drug_shape, latent_dimension),
                self.latent_activation()
            ),
            'target_projector': nn.Sequential(
                nn.Linear(target_shape, latent_dimension),
                self.latent_activation()
            ),
            'drug_residual_block': nn.Sequential(
                *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
            ),
            'target_residual_block': nn.Sequential(
                *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
            )
        })

    def create_simple_network(self, drug_shape, target_shape, latent_dimension):
        # 创建一个简单子网络，包括药物投影和靶标投影（不含残差块）
        return nn.ModuleDict({
            'drug_projector': nn.Sequential(
                nn.Linear(drug_shape, latent_dimension),
                self.latent_activation()
            ),
            'target_projector': nn.Sequential(
                nn.Linear(target_shape, latent_dimension),
                self.latent_activation()
            )
        })

    def forward(self, drug, target, target1, target2):
        if self.do_classify:
            return self.classify(drug, target, target1, target2)
        else:
            return self.regress(drug, target, target1, target2)

    def regress(self, drug, target, target1, target2):
        inner_prods = []

        # 处理第一个目标
        network = self.networks[0]
        drug_projection = network['drug_projector'](drug)
        drug_projection = network['drug_residual_block'](drug_projection)

        target_projection = network['target_projector'](target)
        target_projection = network['target_residual_block'](target_projection)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        inner_prods.append(inner_prod)

        # 处理第二个和第三个目标
        for i, (target_data, network) in enumerate(zip([target1, target2], self.networks[1:])):
            drug_projection = network['drug_projector'](drug)  # 不使用残差块
            target_projection = network['target_projector'](target_data)

            inner_prod = torch.bmm(
                drug_projection.view(-1, 1, self.latent_dimension),
                target_projection.view(-1, self.latent_dimension, 1),
            ).squeeze()
            inner_prods.append(inner_prod)

        return inner_prods

    def classify(self, drug, target, target1, target2):
        distances = []

        # 处理第一个目标
        network = self.networks[0]
        drug_projection = network['drug_projector'](drug)
        drug_projection = network['drug_residual_block'](drug_projection)

        target_projection = network['target_projector'](target)
        target_projection = network['target_residual_block'](target_projection)

        distance = self.activator(drug_projection, target_projection)
        distances.append(distance)

        # 处理第二个和第三个目标
        for i, (target_data, network) in enumerate(zip([target1, target2], self.networks[1:])):
            drug_projection = network['drug_projector'](drug)  # 不使用残差块
            target_projection = network['target_projector'](target_data)

            distance = self.activator(drug_projection, target_projection)
            distances.append(distance)

        return distances

    

# class SimpleCoembeddingResnet2(nn.Module):
#     def __init__(
#         self,
#         drug_shape=2048,
#         target_shape=1024,
#         target_shape_1=1024,
#         # target_shape_2=1024,
#         latent_dimension=1024,
#         latent_activation="ReLU",
#         latent_distance="Cosine",
#         classify=True,
#         num_residual_blocks=2,  # 添加残差块的数量
#         dropout_rate=0.5,  # Dropout 比例
#     ):
#         super().__init__()
#         self.drug_shape = drug_shape
#         self.target_shape = target_shape
#         self.target_shape_1 = target_shape_1
#         # self.target_shape_2 = target_shape_2
#         self.latent_dimension = latent_dimension
#         self.do_classify = classify
#         self.latent_activation = ACTIVATIONS[latent_activation]

#         # 药物投影
#         self.drug_projector = nn.Sequential(
#             nn.Linear(self.drug_shape, latent_dimension),
#             self.latent_activation()
#         )
#         nn.init.xavier_normal_(self.drug_projector[0].weight)

#         # 添加药物的残差块
#         self.drug_residual_blocks = nn.Sequential(
#             *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
#         )

#         # 靶标投影
#         self.target_projectors = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(self.target_shape, latent_dimension),
#                 self.latent_activation()
#             ),
#             nn.Sequential(
#                 nn.Linear(self.target_shape_1, latent_dimension),
#                 self.latent_activation()
#             ),
#             # nn.Sequential(
#             #     nn.Linear(self.target_shape_2, latent_dimension),
#             #     self.latent_activation()
#             # )
#         ])
#         for projector in self.target_projectors:
#             nn.init.xavier_normal_(projector[0].weight)

#         # 添加靶标的残差块
#         self.target_residual_blocks = nn.ModuleList([
#             nn.Sequential(
#                 *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
#             ) for _ in range(2)
#         ])

#         if self.do_classify:
#             self.distance_metric = latent_distance
#             self.activator = DISTANCE_METRICS[self.distance_metric]()

#     def forward(self, drug, target, target1):
#         if self.do_classify:
#             return self.classify(drug, target, target1)
#         else:
#             return self.regress(drug, target, target1)

#     def regress(self, drug, target, target1):
#         drug_projection = self.drug_projector(drug)
#         drug_projection = self.drug_residual_blocks(drug_projection)

#         target_projections = []
#         for i, (target, projector, residual_blocks) in enumerate(zip([target, target1], self.target_projectors, self.target_residual_blocks)):
#             target_projection = projector(target)
#             target_projection = residual_blocks(target_projection)
#             target_projections.append(target_projection)

#         inner_prods = []
#         for target_projection in target_projections:
#             inner_prod = torch.bmm(
#                 drug_projection.view(-1, 1, self.latent_dimension),
#                 target_projection.view(-1, self.latent_dimension, 1),
#             ).squeeze()
#             inner_prods.append(inner_prod)

#         return inner_prods

#     def classify(self, drug, target, target1):
#         drug_projection = self.drug_projector(drug)
#         drug_projection = self.drug_residual_blocks(drug_projection)

#         target_projections = []
#         for i, (target, projector, residual_blocks) in enumerate(zip([target, target1], self.target_projectors, self.target_residual_blocks)):
#             target_projection = projector(target)
#             target_projection = residual_blocks(target_projection)
#             target_projections.append(target_projection)

#         distances = []
#         for target_projection in target_projections:
#             distance = self.activator(drug_projection, target_projection)
#             distances.append(distance)

#         return distances


class SimpleCoembeddingResnet2(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        target_shape_1=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        num_residual_blocks=2,  # 添加残差块的数量
        dropout_rate=0.5,  # Dropout 比例
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.target_shape_1 = target_shape_1
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        # 药物投影
        self.drug_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.drug_shape, latent_dimension),
                self.latent_activation()
            ) for _ in range(2)
        ])
        for i, projector in enumerate(self.drug_projectors):
            projector[0].name = f"drug_projector_{i}_linear"

        nn.init.xavier_normal_(self.drug_projectors[0][0].weight)

        # 添加药物的残差块（仅第一个药物）
        self.drug_residual_block = nn.Sequential(
            *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
        )
        for i, block in enumerate(self.drug_residual_block):
            block.name = f"drug_residual_block_{i}"

        # 靶标投影
        self.target_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.target_shape, latent_dimension),
                self.latent_activation()
            ),
            nn.Sequential(
                nn.Linear(self.target_shape_1, latent_dimension),
                self.latent_activation()
            )
        ])
        for i, projector in enumerate(self.target_projectors):
            projector[0].name = f"target_projector_{i}_linear"

        for projector in self.target_projectors:
            nn.init.xavier_normal_(projector[0].weight)

        # 添加靶标的残差块（仅第一个目标）
        self.target_residual_block = nn.Sequential(
            *[ResidualBlock(latent_dimension, latent_dimension, self.latent_activation, dropout_rate) for _ in range(num_residual_blocks)]
        )
        for i, block in enumerate(self.target_residual_block):
            block.name = f"target_residual_block_{i}"

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target, target1):
        if self.do_classify:
            return self.classify(drug, target, target1)
        else:
            return self.regress(drug, target, target1)

    def regress(self, drug, target, target1):
        # 处理第一个药物（带残差块）
        drug_projection = self.drug_projectors[0](drug)
        drug_projection = self.drug_residual_block(drug_projection)

        # 处理第一个目标（带残差块）
        target_projection = self.target_projectors[0](target)
        target_projection = self.target_residual_block(target_projection)

        # 处理第二个药物和目标（不带残差块）
        drug_projection_1 = self.drug_projectors[1](drug)
        target_projection_1 = self.target_projectors[1](target1)

        inner_prods = []
        # 计算内积
        inner_prod_0 = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        inner_prods.append(inner_prod_0)

        inner_prod_1 = torch.bmm(
            drug_projection_1.view(-1, 1, self.latent_dimension),
            target_projection_1.view(-1, self.latent_dimension, 1),
        ).squeeze()
        inner_prods.append(inner_prod_1)

        return inner_prods

    def classify(self, drug, target, target1):
        # 处理第一个药物（带残差块）
        drug_projection = self.drug_projectors[0](drug)
        drug_projection = self.drug_residual_block(drug_projection)

        # 处理第一个目标（带残差块）
        target_projection = self.target_projectors[0](target)
        target_projection = self.target_residual_block(target_projection)

        # 处理第二个药物和目标（不带残差块）
        drug_projection_1 = self.drug_projectors[1](drug)
        target_projection_1 = self.target_projectors[1](target1)

        distances = []
        # 计算距离
        distance_0 = self.activator(drug_projection, target_projection)
        distances.append(distance_0)

        distance_1 = self.activator(drug_projection_1, target_projection_1)
        distances.append(distance_1)

        return distances








class SimpleCoembeddingSigmoid(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCoembedding_FoldSeek(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=1024,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(foldseek_indices).mean(dim=1)

        full_target_embedding = torch.cat([plm_embedding, foldseek_embedding], dim=1)
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class SimpleCoembedding_FoldSeekX(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=512,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        # self.projector_dropout = nn.Dropout(p=0.2)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(foldseek_indices).mean(dim=1)

        full_target_embedding = torch.cat([plm_embedding, foldseek_embedding], dim=1)
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class GoldmanCPI(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=100,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        model_dropout=0.2,
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        self.last_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, 1, bias=True),
        )

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)
        output = torch.einsum("bd,bd->bd", drug_projection, target_projection)
        distance = self.last_layers(output)
        return distance

    def classify(self, drug, target):
        distance = self.regress(drug, target)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class AffinityCoembedInner(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.mol_projector[0].weight)

        print(self.mol_projector[0].weight)

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.prot_projector[0].weight)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        print(mol_proj)
        print(prot_proj)
        y = torch.bmm(
            mol_proj.view(-1, 1, self.latent_size),
            prot_proj.view(-1, self.latent_size, 1),
        ).squeeze()
        return y


class CosineBatchNorm(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, self.latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, self.latent_size),
            latent_activation(),
        )

        self.mol_norm = nn.BatchNorm1d(self.latent_size)
        self.prot_norm = nn.BatchNorm1d(self.latent_size)

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_norm(self.mol_projector(mol_emb))
        prot_proj = self.prot_norm(self.prot_projector(prot_emb))

        return self.activator(mol_proj, prot_proj)


class LSTMCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        lstm_layers=3,
        lstm_dim=256,
        latent_size=256,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.rnn = nn.LSTM(
            self.prot_emb_size,
            lstm_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(2 * lstm_layers * lstm_dim, latent_size), nn.ReLU()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)

        outp, (h_out, _) = self.rnn(prot_emb)
        prot_hidden = h_out.permute(1, 0, 2).reshape(outp.shape[0], -1)
        prot_proj = self.prot_projector(prot_hidden)

        return self.activator(mol_proj, prot_proj)


class DeepCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        hidden_size=4096,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, hidden_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
            nn.Linear(hidden_size, latent_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class SimpleConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        hidden_dim_1=512,
        hidden_dim_2=256,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.fc1 = nn.Sequential(
            nn.Linear(mol_emb_size + prot_emb_size, hidden_dim_1), activation()
        )
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim_1, hidden_dim_2), activation())
        self.fc3 = nn.Sequential(nn.Linear(hidden_dim_2, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc3(self.fc2(self.fc1(cat_emb))).squeeze()


class SeparateConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric=None,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.fc = nn.Sequential(nn.Linear(2 * latent_size, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


class AffinityEmbedConcat(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )

        self.fc = nn.Linear(2 * latent_size, 1)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


SimplePLMModel = AffinityEmbedConcat


class AffinityConcatLinear(nn.Module):
    def __init__(
        self,
        mol_emb_size,
        prot_emb_size,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.fc = nn.Linear(mol_emb_size + prot_emb_size, 1)

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc(cat_emb).squeeze()
