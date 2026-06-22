"""
ViTTT 参数量统计脚本。

功能：
1. 不依赖 mmpretrain、mmengine、mmcv。
2. 直接读取 ViTTT 的 yaml 配置并构建模型。
3. 可选择加载 ViTTT 训练得到的 checkpoint，例如 max_acc.pth。
4. 可统计总参数量、可训练参数量，以及指定关键字模块的参数量。
5. 可模拟冻结骨干，只统计 head 或自定义模块的可训练参数量。

运行示例：
python count_vittt_params.py
"""

from pathlib import Path
import sys
from argparse import Namespace
from typing import Iterable, Sequence

import torch


# ================== 配置区 ==================
# ViTTT 的 vittt 目录，也就是里面有 main_ema.py、config.py、models、data 的目录
VITTT_ROOT = Path(r"/home/bsj/data/ViTTT-master/vittt")

# ViTTT 的配置文件
CONFIG = VITTT_ROOT / "cfgs/h_vittt_b.yaml"

# 可选：如果只想统计模型结构参数量，可以设为 None
# 如果想加载训练后的权重再统计，可以填 max_acc.pth 路径
CHECKPOINT = None
# CHECKPOINT = Path(r"/data/bsj/vittt/bracs/FT/h_vittt_base/default/max_acc.pth")

# 自己数据集的类别数。
# BRACS 七分类填 7，BACH 四分类填 4。
NUM_CLASSES = 7

# 是否模拟冻结骨干。
# False 表示统计全量微调时的可训练参数。
# True 表示先冻结全部参数，再只打开 TRAIN_KEYWORDS 命中的参数。
FREEZE_BACKBONE = True

# 冻结骨干时，哪些参数名保持可训练。
# 只训练分类头：("head",)
# 训练分类头和自己的模块：("head", "mhcl") 或 ("head", "adapter")
TRAIN_KEYWORDS = ("head", "lora")

# 额外统计哪些关键字模块的参数量。
# 例如你自己的模块名字含有 mhcl，就填 ("mhcl",)
# 如果想统计 TTT 相关参数，也可以加 "ttt"。
MODULE_KEYWORDS = ("mhcl", "adapter", "my_module", "ttt")
# ============================================


def add_vittt_to_path(vittt_root: Path) -> None:
    vittt_root = vittt_root.resolve()
    if not vittt_root.exists():
        raise FileNotFoundError(f"VITTT_ROOT 不存在：{vittt_root}")
    if str(vittt_root) not in sys.path:
        sys.path.insert(0, str(vittt_root))


def make_args(cfg_path: Path) -> Namespace:
    """
    构造 get_config 需要的 args。
    这里不启动分布式，也不训练，只用于构建模型。
    """
    return Namespace(
        cfg=str(cfg_path),
        opts=None,
        batch_size=None,
        data_path=None,
        zip=False,
        cache_mode="part",
        resume=None,
        use_checkpoint=False,
        amp=False,
        output="output",
        tag=None,
        eval=False,
        throughput=False,
        pretrained="",
        find_unused_params=False,
        freeze_backbone=False,
        eval_split="val",
        model_ema=True,
        model_ema_force_cpu=False,
        model_ema_decay=0.9996,
    )


def set_num_classes(config, num_classes: int):
    config.defrost()
    config.MODEL.NUM_CLASSES = int(num_classes)
    # 有些 ViTTT 代码会依赖 LOCAL_RANK，这里给一个默认值。
    config.LOCAL_RANK = 0
    config.freeze()
    return config


def load_checkpoint_if_needed(model: torch.nn.Module, checkpoint_path):
    if checkpoint_path is None:
        print("未加载 checkpoint，只统计模型结构参数量。")
        return

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"CHECKPOINT 不存在：{checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 兼容可能存在的 module. 前缀。
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        new_state_dict[key] = value

    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"已加载 checkpoint：{checkpoint_path}")
    print(msg)


def freeze_selected_params(model: torch.nn.Module, train_keywords: Sequence[str]) -> None:
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if any(keyword in name for keyword in train_keywords):
            param.requires_grad = True


def count_params(model: torch.nn.Module, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_keyword_params(
    model: torch.nn.Module,
    keywords: Iterable[str],
    only_trainable: bool = False,
) -> dict:
    result = {}
    for keyword in keywords:
        total = 0
        matched = []
        for name, param in model.named_parameters():
            if keyword.lower() in name.lower():
                if only_trainable and not param.requires_grad:
                    continue
                total += param.numel()
                matched.append((name, param.numel()))
        result[keyword] = {
            "total": total,
            "matched": matched,
        }
    return result


def print_trainable_params(model: torch.nn.Module) -> None:
    print("\n========== Trainable parameters ==========")
    has_trainable = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_trainable = True
            print(f"{name}: {param.numel() / 1e6:.4f} M")
    if not has_trainable:
        print("没有可训练参数。")


def print_keyword_stats(model: torch.nn.Module, keywords: Sequence[str]) -> None:
    print("\n========== Keyword parameter statistics ==========")
    stats_all = count_keyword_params(model, keywords, only_trainable=False)
    stats_train = count_keyword_params(model, keywords, only_trainable=True)

    for keyword in keywords:
        total_all = stats_all[keyword]["total"]
        total_train = stats_train[keyword]["total"]
        if total_all == 0:
            continue
        print(
            f"Keyword '{keyword}': "
            f"all {total_all / 1e6:.4f} M, "
            f"trainable {total_train / 1e6:.4f} M"
        )


def main():
    add_vittt_to_path(VITTT_ROOT)

    from config import get_config
    from models import build_model

    args = make_args(CONFIG)
    config = get_config(args)
    config = set_num_classes(config, NUM_CLASSES)

    print("========== ViTTT config ==========")
    print(f"Config: {CONFIG}")
    print(f"Model type: {config.MODEL.TYPE}")
    print(f"Model name: {config.MODEL.NAME}")
    print(f"Num classes: {config.MODEL.NUM_CLASSES}")

    model = build_model(config)

    print("\n========== Check MHCLA modules ==========")
    found_mhcla_module = False
    for name, module in model.named_modules():
        if "mhcla" in name.lower() or "mhcl" in name.lower():
            found_mhcla_module = True
            print(name, type(module))

    if not found_mhcla_module:
        print("没有在模型结构中找到 mhcla / mhcl 模块。说明当前 build_model 没有构建到你的 adapter。")

    print("\n========== Check MHCLA parameters ==========")
    found_mhcla_param = False
    for name, param in model.named_parameters():
        if "mhcla" in name.lower() or "mhcl" in name.lower():
            found_mhcla_param = True
            print(name, param.numel() / 1e6, "M")

    if not found_mhcla_param:
        print("没有在参数中找到 mhcla / mhcl。")

    load_checkpoint_if_needed(model, CHECKPOINT)

    if FREEZE_BACKBONE:
        freeze_selected_params(model, TRAIN_KEYWORDS)
        print("\n已模拟冻结骨干。")
        print(f"保持可训练的关键字：{TRAIN_KEYWORDS}")
    else:
        print("\n未冻结骨干，统计全量可训练参数。")

    total_params = count_params(model, only_trainable=False)
    trainable_params = count_params(model, only_trainable=True)

    print_trainable_params(model)
    print_keyword_stats(model, MODULE_KEYWORDS)

    print("\n========== Summary ==========")
    print(f"Total parameters: {total_params / 1e6:.4f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.4f} M")
    print(f"Frozen parameters: {(total_params - trainable_params) / 1e6:.4f} M")


if __name__ == "__main__":
    main()
