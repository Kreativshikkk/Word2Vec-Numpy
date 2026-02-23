from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hallucination_detection.dataset import BatchLoader, CustomDataset, load_dict_pickle
from hallucination_detection.main import CONFIG
from hallucination_detection.model import SkipGramModel
from tqdm.auto import tqdm

from hallucination_detection.utils import load_model_from_ckpt

CKPT_DIRECTORY = Path(CONFIG.get("ckpt_directory", Path("data/checkpoints")))
LOAD_FROM_CKPT = CONFIG.get("load_from_ckpt", False)
ckpt_name = Path(CONFIG.get("ckpt_name", ""))

if ckpt_name == "" and LOAD_FROM_CKPT:
    raise ValueError("ckpt_name cannot be an empty string if you want to load from checkpoint")

CKPT_PATH = CKPT_DIRECTORY / ckpt_name

alpha = 0.05


def measure_accuracy(gt: np.ndarray, pred: np.ndarray) -> float:
    b = gt.shape[0]
    c = gt.shape[1]
    acc = 0.0

    for i in range(b):
        inter = np.intersect1d(gt[i], pred[i])
        acc += len(inter) / c

    return acc / b


def eval(_model: SkipGramModel | None, test_path: str = "data/test.csv", ckpt_path: str | Path | None = CKPT_PATH,
         max_steps: int = 100):
    id_2_word = load_dict_pickle("data/id2word.pkl")

    if _model is None and ckpt_path is None:
        raise ValueError("Either _model or ckpt_path must be provided")
    if _model is None:
        _model = SkipGramModel()
        if Path(ckpt_path).exists():
            ckpt = np.load(ckpt_path)
            _model.w_in.embeddings = ckpt["w_in"]
            _model.w_out.embeddings = ckpt["w_out"]
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    dataset = CustomDataset(test_path)
    pbar = tqdm(BatchLoader(dataset))
    ema = None
    acc_values = []
    for step, (center_words, context_words, negative_words) in enumerate(pbar):
        center_words = np.array(center_words, dtype=np.int64)
        context_words = np.array(context_words, dtype=np.int64)
        negative_words = np.array(negative_words, dtype=np.int64)

        loss, cache = _model.forward(center_words, context_words, negative_words)
        ema = loss if step == 0 else ema * (1 - alpha) + loss * alpha

        preds = _model.predict_next_word(center_words)
        acc = measure_accuracy(context_words, preds)
        pbar.set_postfix({"epoch_loss": ema, "accuracy": acc})

        acc_values.append(acc)
        if step > max_steps:
            break
        if step % 10 == 0:
            for i in range(3):
                center_word = center_words[i]
                gt_context = context_words[i]
                pred_context = preds[i]
                print(f"Center word: {id_2_word[center_word]}")
                print("GT context:", [id_2_word[wid] for wid in gt_context])
                print("Predicted context:", [id_2_word[wid] for wid in pred_context])
                print()
    return acc_values


if __name__ == "__main__":
    model = SkipGramModel()
    if LOAD_FROM_CKPT and CKPT_PATH.exists():
        load_model_from_ckpt(model, CKPT_PATH)
    acc_values = eval(model, max_steps=100)
    x = np.arange(len(acc_values))
    fig, ax = plt.subplots()
    ax.set_title("Accuracy over eval dataset")
    ax.plot(x, acc_values)
    fig.savefig("accuracy_plot.png")
