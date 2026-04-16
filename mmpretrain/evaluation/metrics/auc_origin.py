import torch
import numpy as np
import torch.nn.functional as F

from mmengine.evaluator import BaseMetric
from mmpretrain.registry import METRICS
from sklearn.metrics import roc_auc_score
from typing import List, Optional, Sequence, Union
from .single_label import to_tensor


@METRICS.register_module()
class SingleLabelAUC(BaseMetric):

    default_prefix: Optional[str] = 'accuracy'

    def __init__(self,
                 thrs: Union[float, Sequence[Union[float, None]], None] = 0.,
                 average: Optional[str] = 'macro',
                 num_classes: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if isinstance(thrs, float) or thrs is None:
            self.thrs = (thrs, )
        else:
            self.thrs = tuple(thrs)

        self.average = average
        self.num_classes = num_classes

    def process(self, data_batch, data_samples: Sequence[dict]):


        for data_sample in data_samples:
            result = dict()
            if 'pred_score' in data_sample:
                result['pred_score'] = data_sample['pred_score'].cpu()
            else:
                num_classes = self.num_classes or data_sample.get(
                    'num_classes')
                assert num_classes is not None, \
                    'The `num_classes` must be specified if no `pred_score`.'
                result['pred_label'] = data_sample['pred_label'].cpu()
                result['num_classes'] = num_classes
            result['gt_label'] = data_sample['gt_label'].cpu()
            self.results.append(result)

    def compute_metrics(self, results: List):

        metrics = {}

        # concat
        target = torch.cat([res['gt_label'] for res in results])
        if 'pred_score' in results[0]:
            pred = torch.stack([res['pred_score'] for res in results])
            auc = self.calculate(
                pred, target, thrs=self.thrs, average=self.average)

            multi_thrs = len(self.thrs) > 1
            for i, thr in enumerate(self.thrs):
                if multi_thrs:
                    suffix = 'auc_no-thr' if thr is None else f'_thr-{thr:.2f}'
                else:
                    suffix = 'auc'
                print(type(auc), auc)
                for k, v in enumerate(auc):
                    print(k, v)
                    # metrics[str(k)+suffix] = v
                    metrics[suffix] = v
        else:
            pred = torch.cat([res['pred_label'] for res in results])
            auc = self.calculate(
                pred,
                target,
                average=self.average,
                num_classes=results[0]['num_classes'])
            metrics['auc'] = auc

        return metrics

    @staticmethod
    def calculate(
        pred: Union[torch.Tensor, np.ndarray, Sequence],
        target: Union[torch.Tensor, np.ndarray, Sequence],
        thrs: Sequence[Union[float, None]] = (0., ),
        average: Optional[str] = 'macro',
        num_classes: Optional[int] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        average_options = ['micro', 'macro', None]
        assert average in average_options, 'Invalid `average` argument, ' \
            f'please specify from {average_options}.'

        pred = to_tensor(pred)
        target = to_tensor(target).to(torch.int64)
        assert pred.size(0) == target.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target.size(0)}).'

        if pred.ndim == 1:
            assert num_classes is not None, \
                'Please specify the `num_classes` if the `pred` is labels ' \
                'intead of scores.'
            gt_positive = F.one_hot(target.flatten(), num_classes)
            pred_positive = F.one_hot(pred.to(torch.int64), num_classes)
            return roc_auc_score(gt_positive, pred_positive,
                                                average=average)
        else:
            # For pred score, calculate on all thresholds.
            num_classes = pred.size(1)
            pred_score, pred_label = torch.topk(pred, k=1)
            pred_score = pred_score.flatten()
            pred_label = pred_label.flatten()

            gt_positive = F.one_hot(target.flatten(), num_classes)

            results = []
            for thr in thrs:
                pred_positive = F.one_hot(pred_label, num_classes)
                if thr is not None:
                    pred_positive[pred_score <= thr] = 0
                results.append(
                    roc_auc_score(gt_positive, pred_positive,
                                                 average=average))

            return results