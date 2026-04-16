from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import torch, os, numpy as np

@HOOKS.register_module()
class DumpFeatureHook(Hook):
    def __init__(self, out_dir='./base', save_per_file=False, strip_prefix=None):
        self.out_dir = out_dir
        self.save_per_file = save_per_file
        self.strip_prefix = strip_prefix

    def before_test(self, runner):
        os.makedirs(self.out_dir, exist_ok=True)
        self._cur_bucket = []    # 存“当前 iter”收集到的 pre-logits（由 forward hook 填）
        self._feats = []         # 聚合后的 (B, C) 特征，跨 iter 累积
        self._ids = []           # 样本 id
        self._labels = []

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        # 兼容 fc_cls / fc 命名
        head = getattr(model, 'cls_head', None) or getattr(model, 'head', None)
        assert head is not None, '未找到 head/cls_head'
        cls_layer = getattr(head, 'fc_cls', None) or getattr(head, 'fc', None)
        assert cls_layer is not None, '未找到分类层（fc_cls 或 fc）'

        def _hook(mod, inp, out):
            # 线性层输入 = pre-logits；形状可能是 (B*V, C) 或 (B, C)
            x = inp[0].detach().float().cpu()
            # 保险：压成二维
            x = x.flatten(1)
            self._cur_bucket.append(x)

        self._handle = cls_layer.register_forward_hook(_hook)

    def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        # 1) 取出当前 iter 的所有 pre-logits，并清空临时桶
        assert len(self._cur_bucket) > 0, '未收到本 iter 的 pre-logits，请检查 hook 是否生效'
        X_nv = torch.cat(self._cur_bucket, dim=0)  # (N, C)；N 可能等于 B*V
        self._cur_bucket.clear()

        # 2) 计算本 iter 的 B / V，并把多视角聚合成 (B, C)
        B = len(data_batch['data_samples'])
        N = X_nv.shape[0]
        assert N % B == 0, f'本 iter 特征数 {N} 不是 batch 大小 {B} 的整数倍；请检查裁剪/视角设置'
        V = N // B
        X = X_nv.view(B, V, -1).mean(dim=1)       # (B, C)
        self._feats.append(X)

        # 3) 收集 id / label（一条对应一个样本）
        for ds in data_batch['data_samples']:
            meta = ds.metainfo
            sid = meta.get('audio_path') or meta.get('filename') or meta.get('img_path') or meta.get('ori_filename')
            sid = str(sid)
            if self.strip_prefix and sid.startswith(self.strip_prefix):
                sid = sid[len(self.strip_prefix):].lstrip('/\\')
            self._ids.append(sid)
            if hasattr(ds, 'gt_label'):
                self._labels.append(int(ds.gt_label))
            elif hasattr(ds, 'gt_labels'):
                self._labels.append(int(ds.gt_labels.item()))
            else:
                self._labels.append(-1)

    def after_test(self, runner):
        if hasattr(self, '_handle') and self._handle is not None:
            self._handle.remove()

        # 拼接所有 iter 的 (B, C) → (N, C)
        X = torch.cat(self._feats, dim=0).numpy().astype('float32')  # (N, C)
        assert len(self._ids) == X.shape[0], \
            f'特征数 {X.shape[0]} 与样本数 {len(self._ids)} 不一致（请检查聚合逻辑）'

        if self.save_per_file:
            for feat, sid in zip(X, self._ids):
                base = os.path.splitext(sid)[0].replace('\\', '/')
                dst = os.path.join(self.out_dir, base + '_feat.npy')
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                np.save(dst, feat)
            runner.logger.info(f'[DumpPreLogitsHook] saved {len(self._ids)} files under {self.out_dir}')
        else:
            np.save(os.path.join(self.out_dir, 'audio_feats.npy'), X)
            with open(os.path.join(self.out_dir, 'audio_ids.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(self._ids))
            np.save(os.path.join(self.out_dir, 'audio_labels.npy'), np.array(self._labels, dtype='int64'))
            runner.logger.info(f'[DumpPreLogitsHook] saved audio_feats.npy {X.shape} + ids/labels in {self.out_dir}')
