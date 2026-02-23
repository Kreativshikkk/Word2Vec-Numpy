from __future__ import annotations

from pathlib import Path

import numpy as np

from main import CONFIG
from model import SkipGramModel

CKPT_DIRECTORY = Path(CONFIG.get("ckpt_directory", Path("data/checkpoints")))

if not CKPT_DIRECTORY.exists():
    CKPT_DIRECTORY.mkdir(parents=True, exist_ok=True)


def load_model_from_ckpt(_model: SkipGramModel, ckpt_path: str | Path):
    ckpt = np.load(ckpt_path)
    _model.w_in.embeddings = ckpt["w_in"]
    _model.w_out.embeddings = ckpt["w_out"]
    _model.w_in.m = ckpt["m_w_in"]
    _model.w_in.v = ckpt["v_w_in"]
    _model.w_in.t = ckpt["t_in"]
    _model.w_out.m = ckpt["m_w_out"]
    _model.w_out.v = ckpt["v_w_out"]
    _model.w_out.t = ckpt["t_out"]
    print(f"Loaded checkpoint from {ckpt_path}")