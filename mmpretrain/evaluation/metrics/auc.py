
import torch
import numpy as np
import torch.nn.functional as F

from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS
from sklearn.metrics import roc_auc_score
from typing import List, Optional
from .single_label import to_tensor


def _safe_roc_auc_score(y_true, y_score, average=None, **kwargs):
    """Wrapper that returns NaN instead of raising when only one class present."""
    try:
        return roc_auc_score(y_true, y_score, average=average, **kwargs)
    except ValueError as e:
        # Typical message: "Only one class present in y_true. ROC AUC score is not defined in that case."
        if "Only one class present" in str(e):
            return float("nan")
        raise


@METRICS.register_module()
class SingleLabelAUC(BaseMetric):
    """Drop-in replacement that matches the original interface but prevents crashes
    when a class is single-valued in the evaluation slice.
    """

    default_prefix: Optional[str] = 'accuracy'

    def __init__(self,
                 thrs: Optional[List[float]] = None,
                 average: str = 'macro',
                 num_classes: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.thrs = thrs if thrs is not None else [None]
        self.average = average
        self.num_classes = num_classes

    @staticmethod
    def calculate(pred,
                  target,
                  thrs=(None,),
                  average='macro',
                  num_classes: Optional[int] = None):
        """Compute AUC with minimal changes to original logic, but safely.

        - Keeps original behavior of using one-hot labels
        - Iterates thresholds if provided
        - Returns NaN for classes/columns without both positive and negative samples
        """
        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)

        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match the target ({target.size(0)})."

        if pred.ndim == 1:
            # pred is label indices; need num_classes
            assert num_classes is not None, \
                'Please specify `num_classes` if `pred` are labels instead of scores.'
            gt_positive = F.one_hot(target.flatten(), num_classes).cpu().numpy()
            pred_positive = F.one_hot(pred.to(torch.int64), num_classes).cpu().numpy()
            return _safe_roc_auc_score(gt_positive, pred_positive, average=average)

        # pred is [N, C] scores (logits/probs); original code used top-1 labels + thresholding.
        # To keep changes minimal, we preserve that flow but guard AUC.
        num_classes = pred.size(1) if num_classes is None else num_classes
        pred_score = pred.max(dim=1).values.flatten().cpu().numpy()
        pred_label = pred.argmax(dim=1).to(torch.int64).flatten()

        gt_positive = F.one_hot(target.flatten(), num_classes).cpu().numpy()

        results = []
        for thr in thrs:
            pred_positive = F.one_hot(pred_label, num_classes).cpu().numpy()
            if thr is not None:
                # Zero-out predictions whose confidence is below threshold
                mask = pred_score <= thr
                if mask.any():
                    pred_positive[mask] = 0
            results.append(_safe_roc_auc_score(gt_positive, pred_positive, average=average))
        return results

    def process(self, data_batch, data_samples):
        # mmengine requires this; we simply pass through to collect predictions
        for sample in data_samples:
            self.results.append({
                'pred_score': to_tensor(sample.get('pred_score')) if 'pred_score' in sample else None,
                'pred_label': to_tensor(sample.get('pred_label')) if 'pred_label' in sample else None,
                'gt_label': to_tensor(sample.get('gt_label'))
            })

    def compute_metrics(self, results):
        # Aggregate predictions and call calculate like the original implementation.
        # Prefer pred_score if available; else fall back to pred_label.
        has_score = all(r.get('pred_score') is not None for r in results)
        num_classes = self.num_classes
        if has_score:
            pred = torch.stack([r['pred_score'] for r in results], dim=0)
        else:
            pred = torch.stack([r['pred_label'] for r in results], dim=0)

        target = torch.stack([r['gt_label'] for r in results], dim=0)

        auc = self.calculate(pred, target, thrs=tuple(self.thrs), average=self.average, num_classes=num_classes)
        # Normalize the return shape to a dict
        if isinstance(auc, list):
            # when thresholds provided
            return {f'auc@thr={thr}': v for thr, v in zip(self.thrs, auc)}
        return {'auc': auc}
