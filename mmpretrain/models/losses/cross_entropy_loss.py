# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from mmpretrain.registry import MODELS
from .utils import weight_reduce_loss


def cross_entropy(pred,
                  label,
                  weight=None,
                  reduction='mean',
                  avg_factor=None,
                  class_weight=None):
    """Calculate the CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = F.cross_entropy(pred, label, weight=class_weight, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def soft_cross_entropy(pred,
                       label,
                       weight=None,
                       reduction='mean',
                       class_weight=None,
                       avg_factor=None):
    """Calculate the Soft CrossEntropy loss. The label can be float.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        label (torch.Tensor): The gt label of the prediction with shape (N, C).
            When using "mixup", the label can be float.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str): The method used to reduce the loss.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # element-wise losses
    loss = -label * F.log_softmax(pred, dim=-1)
    if class_weight is not None:
        loss *= class_weight
    loss = loss.sum(dim=-1)

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         pos_weight=None):
    r"""Calculate the binary CrossEntropy loss with logits.

    Args:
        pred (torch.Tensor): The prediction with shape (N, \*).
        label (torch.Tensor): The gt label with shape (N, \*).
        weight (torch.Tensor, optional): Element-wise weight of loss with shape
            (N, ). Defaults to None.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". If reduction is 'none' , loss
            is same shape as pred and label. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (torch.Tensor, optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (torch.Tensor, optional): The positive weight for each
            class with shape (C), C is the number of classes. Default None.

    Returns:
        torch.Tensor: The calculated loss
    """
    # Ensure that the size of class_weight is consistent with pred and label to
    # avoid automatic boracast,
    assert pred.dim() == label.dim()

    if class_weight is not None:
        N = pred.size()[0]
        class_weight = class_weight.repeat(N, 1)
    loss = F.binary_cross_entropy_with_logits(
        pred,
        label.float(),  # only accepts float type tensor
        weight=class_weight,
        pos_weight=pos_weight,
        reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        assert weight.dim() == 1
        weight = weight.float()
        if pred.dim() > 1:
            weight = weight.reshape(-1, 1)
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    """Cross entropy loss.

    Args:
        use_sigmoid (bool): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_soft (bool): Whether to use the soft version of CrossEntropyLoss.
            Defaults to False.
        reduction (str): The method used to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float):  Weight of the loss. Defaults to 1.0.
        class_weight (List[float], optional): The weight for each class with
            shape (C), C is the number of classes. Default None.
        pos_weight (List[float], optional): The positive weight for each
            class with shape (C), C is the number of classes. Only enabled in
            BCE loss when ``use_sigmoid`` is True. Default None.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_soft=False,
                 reduction='mean',
                 loss_weight=1.0,
                 class_weight=None,
                 pos_weight=None):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.use_soft = use_soft
        assert not (
            self.use_soft and self.use_sigmoid
        ), 'use_sigmoid and use_soft could not be set simultaneously'

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.pos_weight = pos_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_soft:
            self.cls_criterion = soft_cross_entropy
        else:
            self.cls_criterion = cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # only BCE loss has pos_weight
        if self.pos_weight is not None and self.use_sigmoid:
            pos_weight = cls_score.new_tensor(self.pos_weight)
            kwargs.update({'pos_weight': pos_weight})
        else:
            pos_weight = None

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

# @MODELS.register_module()
# class AdaptiveCrossEntropyLoss(BaseWeightedLoss):
#     """动态自适应交叉熵损失，集成类别平衡与焦点损失
#
#     Args:
#         loss_weight (float): 损失权重，默认1.0
#         num_classes (int): 数据集的类别总数
#         beta (float): 初始类别平衡因子，默认0.9999
#         gamma (float): 初始焦点调节因子，默认2.0
#         momentum (float): 类别统计动量，默认0.99
#         smooth (float): 平滑系数，默认1e-6
#         auto_params (bool): 是否自动调整beta/gamma，默认True
#         class_weight (Optional[List[float]]): 可选静态类别权重
#     """
#
#     def __init__(self,
#                  loss_weight: float = 1.0,
#                  num_classes: int = 2,  # 必须指定类别总数
#                  beta: float = 0.9999,
#                  gamma: float = 2.0,
#                  momentum: float = 0.99,
#                  smooth: float = 1e-6,
#                  auto_params: bool = True,
#                  class_weight: Optional[List[float]] = None) -> None:
#         super().__init__(loss_weight=loss_weight)
#         if num_classes is None:
#             raise ValueError("必须指定num_classes参数")
#
#         self.beta = beta
#         self.base_gamma = gamma
#         self.momentum = momentum
#         self.smooth = smooth
#         self.auto_params = auto_params
#         self.num_classes = num_classes
#
#         # 原始class_weight兼容
#         self.class_weight = None
#         if class_weight is not None:
#             assert len(class_weight) == num_classes, "class_weight长度需与num_classes一致"
#             self.class_weight = torch.Tensor(class_weight)
#
#         # 初始化统计参数
#         self.register_buffer('class_counts', torch.zeros(num_classes))
#         self.register_buffer('global_steps', torch.tensor(0))
#         self.register_buffer('initialized', torch.tensor(True))  # 取消动态初始化
#
#     def update_class_stats(self, targets):
#         """动量更新类别分布统计"""
#         # 当前batch统计（替换bincount为unique+手动填充）
#         unique_labels, counts = torch.unique(targets, return_counts=True)
#         batch_counts = torch.zeros(self.num_classes, device=targets.device, dtype=torch.float)
#         batch_counts[unique_labels] = counts.float()
#
#         # 动量更新
#         self.class_counts = self.momentum * self.class_counts + (1 - self.momentum) * batch_counts
#         self.global_steps += 1
#
#     def compute_adaptive_weights(self, device):
#         """计算动态类别权重"""
#         freq = self.class_counts / (self.class_counts.sum() + self.smooth)
#
#         # 有效样本量公式
#         effective_num = 1.0 - torch.pow(self.beta, self.class_counts + self.smooth)
#         weights = (1.0 - self.beta) / (effective_num + self.smooth)
#
#         # 归一化处理
#         return weights / weights.sum() * len(weights)
#
#     def compute_dynamic_gamma(self):
#         """动态调整gamma值"""
#         if not self.auto_params:
#             return self.base_gamma
#
#         # 基于训练进度调整
#         progress = torch.sigmoid(torch.tensor(self.global_steps / 1000.0 - 2.0))
#         return self.base_gamma * (1.0 + 0.5 * progress)
#
#     def _forward(self, cls_score: torch.Tensor, label: torch.Tensor, **kwargs) -> torch.Tensor:
#         # 更新类别统计
#         if self.training:
#             self.update_class_stats(label)
#
#         # 计算动态参数
#         gamma = self.compute_dynamic_gamma()
#         weights = self.compute_adaptive_weights(cls_score.device)
#
#         # 基础交叉熵计算
#         if cls_score.size() == label.size():
#             # 软标签处理（保持原逻辑）
#             lsm = F.log_softmax(cls_score, 1)
#             if self.class_weight is not None:
#                 lsm = lsm * self.class_weight.to(cls_score.device).unsqueeze(0)
#             ce_loss = -(label * lsm).sum(1)
#         else:
#             # 硬标签增强处理
#             ce_loss = F.cross_entropy(cls_score, label, reduction='none')
#
#         # 焦点损失调制
#         pt = torch.exp(-ce_loss.detach())  # 分离计算图
#         focal_weight = (1 - pt) ** gamma
#         weighted_loss = focal_weight * ce_loss
#
#         # 应用动态类别权重
#         if self.class_weight is None:  # 优先使用动态权重
#             weighted_loss = weights[label] * weighted_loss
#
#         # 归约处理
#         return weighted_loss.mean()
