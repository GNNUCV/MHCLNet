import sys
from pathlib import Path
from collections import defaultdict, Counter
import csv
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm


# ================== 配置区 ==================
# ViTTT 的 vittt 目录，也就是里面有 main_ema.py、config.py、models、data 的目录
VITTT_ROOT = Path(r"/vittt")

# ViTTT 的配置文件
CONFIG = r"/home/bsj/data/ViTTT-master/vittt/cfgs/h_vittt_b.yaml"

# 训练后保存的最佳验证集权重，建议用 max_acc.pth，不要用 max_ema_acc.pth
CHECKPOINT = r"/data/bsj/vittt/bach/hccr/h_vittt_base/default/max_acc.pth"
out_dir = r"/data/bsj/vittt/bach/hccr/h_vittt_base/default"

# BACH patch 测试集目录
# 目录结构应为：
# TEST_ROOT/Benign/*.tif
# TEST_ROOT/InSitu/*.tif
# TEST_ROOT/Invasive/*.tif
# TEST_ROOT/Normal/*.tif
TEST_ROOT = Path(r"/home/bsj/data/BACH_color_split_7_3_clean/test")

# 类别顺序必须和训练时 ImageFolder 的类别顺序一致。
# torchvision.datasets.ImageFolder 默认按文件夹名字母排序。
CLASSES = ["Benign", "InSitu", "Invasive", "Normal"]

DEVICE = "cuda:0"
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
# ============================================


# 让脚本能直接导入 ViTTT 的 config、models、data
if VITTT_ROOT.exists():
    sys.path.insert(0, str(VITTT_ROOT))
else:
    # 如果你把脚本放在 vittt 目录下运行，这里也能兼容
    sys.path.insert(0, str(Path(__file__).resolve().parent))

sys.path.insert(0,str(VITTT_ROOT))
from config  import get_config
from models import build_model
from data.build import build_transform


# b091_r00_c01.tif -> b091
# b091_r00_c01.png -> b091
pattern = re.compile(r"(.+)_r\d{2}_c\d{2}$", re.IGNORECASE)


def make_vittt_args():
    """
    构造 get_config(args) 需要的最小参数。
    这里不训练，只用于构建模型和测试 transform。
    """
    class Args:
        pass

    args = Args()
    args.cfg = CONFIG
    args.opts = None
    args.batch_size = None
    args.data_path = str(TEST_ROOT.parent)
    args.zip = False
    args.cache_mode = "part"
    args.resume = None
    args.use_checkpoint = False
    args.amp = False
    args.output = "output"
    args.tag = None
    args.eval = False
    args.throughput = False
    return args


def build_vittt_model(config, checkpoint_path, device):
    """
    构建 ViTTT 模型，并加载训练好的 checkpoint。
    兼容两种情况：
    1. 你的 build_model 已经支持 config.MODEL.NUM_CLASSES。
    2. build_model 仍然默认 1000 类，这里会手动替换 head。
    """
    config.defrost()
    config.MODEL.NUM_CLASSES = len(CLASSES)
    config.freeze()

    model = build_model(config)

    # 如果模型分类头不是 BACH 四分类，就手动替换成四分类。
    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        if model.head.out_features != len(CLASSES):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, len(CLASSES))
            print(f"已将分类头替换为 {len(CLASSES)} 类：Linear({in_features}, {len(CLASSES)})")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # 兼容可能带有 module. 前缀的权重
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print("权重加载信息：", msg)

    model.to(device)
    model.eval()
    return model


def get_image_id(filename):
    stem = Path(filename).stem
    m = pattern.match(stem)
    if m is None:
        return stem
    return m.group(1)


def predict_one_patch(model, transform, img_path, device):
    image = Image.open(img_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device, non_blocking=True)

    with torch.no_grad():
        output = model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        prob = torch.softmax(output, dim=1)[0].detach().cpu().numpy()

    pred_label = int(np.argmax(prob))
    return pred_label, prob


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def build_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def compute_metrics(y_true, y_pred, class_names):
    """
    不依赖 sklearn，直接计算 accuracy、macro precision、macro recall、macro f1。
    同时返回每个类别的 precision、recall、f1、support。
    """
    num_classes = len(class_names)
    cm = build_confusion_matrix(y_true, y_pred, num_classes)

    total = cm.sum()
    correct = np.trace(cm)
    accuracy = safe_div(correct, total) * 100

    per_class = []
    precisions = []
    recalls = []
    f1s = []

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)

        per_class.append({
            "class": name,
            "precision": precision * 100,
            "recall": recall * 100,
            "f1": f1 * 100,
            "support": int(support),
        })

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions)) * 100
    macro_recall = float(np.mean(recalls)) * 100
    macro_f1 = float(np.mean(f1s)) * 100

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm,
    }


def save_confusion_matrix_png(cm, class_names, out_path, title):
    """
    保存混淆矩阵图片。
    行是真实类别，列是预测类别。
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=200)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_metrics(f, title, metrics):
    f.write(f"===== {title} =====\n")
    f.write(f"Accuracy: {metrics['accuracy']:.2f}%\n")
    f.write(f"Macro Precision: {metrics['macro_precision']:.2f}%\n")
    f.write(f"Macro Recall: {metrics['macro_recall']:.2f}%\n")
    f.write(f"Macro F1-score: {metrics['macro_f1']:.2f}%\n")
    f.write("\nPer-class metrics:\n")
    f.write("class,precision,recall,f1-score,support\n")
    for item in metrics["per_class"]:
        f.write(
            f"{item['class']},{item['precision']:.2f},{item['recall']:.2f},"
            f"{item['f1']:.2f},{item['support']}\n"
        )
    f.write("\nConfusion matrix, rows=true, cols=pred:\n")
    f.write(str(metrics["confusion_matrix"]))
    f.write("\n\n")


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    args = make_vittt_args()
    config = get_config(args)

    # 使用 ViTTT 官方验证阶段的 transform。
    # 这个和 data/build.py 里 val/test 的预处理保持一致。
    transform = build_transform(is_train=False, config=config)

    model = build_vittt_model(config, CHECKPOINT, device)

    groups = defaultdict(list)

    # 收集 test 里的所有 patch
    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = TEST_ROOT / cls_name
        if not cls_dir.exists():
            print(f"警告：找不到目录 {cls_dir}")
            continue

        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            image_id = get_image_id(img_path.name)
            key = f"{cls_name}/{image_id}"
            groups[key].append((img_path, cls_idx, cls_name, image_id))

    print(f"test 中共有 {len(groups)} 张原图参与投票")

    patch_y_true = []
    patch_y_pred = []

    img_y_true = []
    img_y_pred_majority = []
    img_y_pred_prob_mean = []

    details = []

    for key, items in tqdm(groups.items()):
        patch_pred_labels = []
        patch_probs = []
        patch_names = []

        true_label = items[0][1]
        true_class = items[0][2]
        image_id = items[0][3]

        for img_path, _, _, _ in items:
            pred_label, prob = predict_one_patch(model, transform, img_path, device)

            patch_pred_labels.append(pred_label)
            patch_probs.append(prob)
            patch_names.append(img_path.name)

            patch_y_true.append(true_label)
            patch_y_pred.append(pred_label)

        # 方式1：多数投票。
        # 如果票数相同，用平均概率作为辅助决策。
        vote_counter = Counter(patch_pred_labels)
        max_vote = max(vote_counter.values())
        candidates = [label for label, count in vote_counter.items() if count == max_vote]

        mean_prob = np.mean(np.stack(patch_probs, axis=0), axis=0)
        if len(candidates) == 1:
            majority_label = candidates[0]
        else:
            majority_label = max(candidates, key=lambda x: mean_prob[x])

        # 方式2：平均概率投票。
        prob_mean_label = int(np.argmax(mean_prob))

        img_y_true.append(true_label)
        img_y_pred_majority.append(majority_label)
        img_y_pred_prob_mean.append(prob_mean_label)

        details.append({
            "image_key": key,
            "image_id": image_id,
            "true_class": true_class,
            "num_patches_in_test": len(items),
            "majority_pred": CLASSES[majority_label],
            "prob_mean_pred": CLASSES[prob_mean_label],
            "patch_files": "|".join(patch_names),
            "patch_preds": "|".join(CLASSES[i] for i in patch_pred_labels),
            "mean_prob": "|".join(f"{x:.6f}" for x in mean_prob.tolist()),
        })

    # 计算指标
    patch_metrics = compute_metrics(patch_y_true, patch_y_pred, CLASSES)
    majority_metrics = compute_metrics(img_y_true, img_y_pred_majority, CLASSES)
    prob_mean_metrics = compute_metrics(img_y_true, img_y_pred_prob_mean, CLASSES)

    print()
    print(f"Patch-level Accuracy: {patch_metrics['accuracy']:.2f}%")
    print(f"Patch-level Macro Precision: {patch_metrics['macro_precision']:.2f}%")
    print(f"Patch-level Macro Recall: {patch_metrics['macro_recall']:.2f}%")
    print(f"Patch-level Macro F1-score: {patch_metrics['macro_f1']:.2f}%")
    print()
    print(f"Majority Voting Image-level Accuracy: {majority_metrics['accuracy']:.2f}%")
    print(f"Majority Voting Image-level Macro Precision: {majority_metrics['macro_precision']:.2f}%")
    print(f"Majority Voting Image-level Macro Recall: {majority_metrics['macro_recall']:.2f}%")
    print(f"Majority Voting Image-level Macro F1-score: {majority_metrics['macro_f1']:.2f}%")
    print()
    print(f"Mean Probability Voting Image-level Accuracy: {prob_mean_metrics['accuracy']:.2f}%")
    print(f"Mean Probability Voting Image-level Macro Precision: {prob_mean_metrics['macro_precision']:.2f}%")
    print(f"Mean Probability Voting Image-level Macro Recall: {prob_mean_metrics['macro_recall']:.2f}%")
    print(f"Mean Probability Voting Image-level Macro F1-score: {prob_mean_metrics['macro_f1']:.2f}%")
    print()


    metrics_file = out_dir / "bach_vittt_vote_metrics.txt"
    details_file = out_dir / "bach_vittt_vote_details.csv"

    # 保存指标文本
    with open(metrics_file, "w", encoding="utf-8") as f:
        write_metrics(f, "Patch-level", patch_metrics)
        write_metrics(f, "Majority Voting Image-level", majority_metrics)
        write_metrics(f, "Mean Probability Voting Image-level", prob_mean_metrics)

    # 保存投票细节 CSV
    with open(details_file, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image_key",
            "image_id",
            "true_class",
            "num_patches_in_test",
            "majority_pred",
            "prob_mean_pred",
            "patch_files",
            "patch_preds",
            "mean_prob",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(details)

    # 保存混淆矩阵图片
    save_confusion_matrix_png(
        patch_metrics["confusion_matrix"],
        CLASSES,
        out_dir / "bach_vittt_patch_confusion_matrix.png",
        "Patch-level Confusion Matrix",
    )
    save_confusion_matrix_png(
        majority_metrics["confusion_matrix"],
        CLASSES,
        out_dir / "bach_vittt_majority_vote_confusion_matrix.png",
        "Majority Voting Image-level Confusion Matrix",
    )
    save_confusion_matrix_png(
        prob_mean_metrics["confusion_matrix"],
        CLASSES,
        out_dir / "bach_vittt_prob_mean_vote_confusion_matrix.png",
        "Mean Probability Voting Image-level Confusion Matrix",
    )

    print(f"指标结果已保存到：{metrics_file}")
    print(f"投票细节已保存到：{details_file}")
    print(f"混淆矩阵图片已保存到：{out_dir}")


if __name__ == "__main__":
    main()
