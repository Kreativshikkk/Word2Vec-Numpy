from __future__ import annotations

from pathlib import Path

import numpy as np

from hallucination_detection.dataset import BatchLoader, CustomDataset
from hallucination_detection.main import CONFIG
from hallucination_detection.model import SkipGramModel
from tqdm.auto import tqdm

from hallucination_detection.utils import load_model_from_ckpt

NUM_EPOCHS = CONFIG.get("num_epochs", 2)
CKPT_DIRECTORY = Path(CONFIG.get("ckpt_directory", Path("data/checkpoints")))
LOAD_FROM_CKPT = CONFIG.get("load_from_ckpt", False)
ckpt_name = Path(CONFIG.get("ckpt_name", ""))

if ckpt_name == "" and LOAD_FROM_CKPT:
    raise ValueError("ckpt_name cannot be an empty string if you want to load from checkpoint")

CKPT_PATH = CKPT_DIRECTORY / ckpt_name


alpha = 0.05


def train(_model: SkipGramModel, train_path: str = "data/train.csv", last_iter: bool | int = False):
    for epoch in range(NUM_EPOCHS):
        ema = None

        pbar = tqdm(BatchLoader(CustomDataset(train_path), last_iter=last_iter))
        for step, (center_words, context_words, negative_words) in enumerate(pbar):
            center_words = np.array(center_words, dtype=np.int64)
            context_words = np.array(context_words, dtype=np.int64)
            negative_words = np.array(negative_words, dtype=np.int64)
            loss, cache = _model.forward(center_words, context_words, negative_words)
            _model.backward(cache)
            ema = loss if ema is None else ema * (1 - alpha) + loss * alpha
            pbar.set_postfix({"epoch_loss": ema})
            if step % 5000 == 0:
                print("p_pos mean", cache["proba_pos"].mean(), "p_neg mean", cache["proba_neg"].mean())
            if step % 1000000 == 0 and step != 0:
                ckpt_path = CKPT_DIRECTORY / f"epoch_{epoch}_step_{step}.npz"
                np.savez(ckpt_path, w_in=model.w_in.embeddings, w_out=model.w_out.embeddings, m_w_in=model.w_in.m,
                         v_w_in=model.w_in.v, t_in=model.w_in.t, m_w_out=model.w_out.m, v_w_out=model.w_out.v,
                         t_out=model.w_out.t)
                print("SAVED ckpt")


if __name__ == "__main__":
    model = SkipGramModel()
    if LOAD_FROM_CKPT and CKPT_PATH.exists():
        load_model_from_ckpt(model, CKPT_PATH)
    train(model)
