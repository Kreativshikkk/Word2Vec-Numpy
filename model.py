from __future__ import annotations

import numpy as np

from hallucination_detection.dataset import load_dict_pickle
from hallucination_detection.main import CONFIG

VOCAB_SIZE = len(load_dict_pickle("data/word2id.pkl"))
EMBEDDING_DIM = CONFIG.get("embedding_dim", 256)
BETA_1 = CONFIG.get("beta_1", 0.9)
BETA_2 = CONFIG.get("beta_2", 0.999)
EPS = float(CONFIG.get("eps", 1e-12))
LR = float(CONFIG.get("learning_rate", 1e-3))


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


class EmbeddingLayer:
    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM, m: np.ndarray | None = None,
                 v: np.ndarray | None = None, t: int = 0):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = np.random.default_rng().normal(loc=0.0, scale=0.01, size=(vocab_size, embedding_dim)).astype(
            np.float32)
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.eps = EPS
        if m is not None and v is not None:
            self.m = m
            self.v = v
            self.t = t
        else:
            self.m = np.zeros_like(self.embeddings)
            self.v = np.zeros_like(self.embeddings)
            self.t = 0

    def forward(self, input_indices: int | np.ndarray):
        return self.embeddings[input_indices]

    def backward(self, input_indices: int | np.ndarray, grad_value: np.ndarray, lr: float | np.float32):
        # bug: update using np.add to accumulate gradients
        idx_flat = np.reshape(input_indices, (-1,))
        grad_flat = np.reshape(grad_value, (-1, self.embedding_dim))

        unique_idx, reverse = np.unique(idx_flat, return_inverse=True)
        grad_sum = np.zeros((len(unique_idx), self.embedding_dim), dtype=grad_flat.dtype)
        np.add.at(grad_sum, reverse, grad_flat)

        self.t += 1
        self.m[unique_idx] = self.m[unique_idx] * self.beta_1 + grad_sum * (1 - self.beta_1)
        self.v[unique_idx] = self.v[unique_idx] * self.beta_2 + (grad_sum ** 2) * (1 - self.beta_2)
        m_hat = self.m[unique_idx] / (1 - self.beta_1 ** self.t)
        v_hat = self.v[unique_idx] / (1 - self.beta_2 ** self.t)
        self.embeddings[unique_idx] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


class SkipGramModel:
    def __init__(self, vocab_size: int = VOCAB_SIZE, embedding_dim: int = EMBEDDING_DIM, lr: float = LR):
        self.w_in = EmbeddingLayer(vocab_size, embedding_dim)
        self.w_out = EmbeddingLayer(vocab_size, embedding_dim)
        self.lr = lr
        self.eps = EPS

    def forward(self, center_word: np.ndarray, context_words: np.ndarray, negative_words: np.ndarray):
        v = self.w_in.forward(center_word)  # (B, D,)
        u_pos = self.w_out.forward(context_words)  # (B, m, D)
        u_neg = self.w_out.forward(negative_words)  # (B, k, D)

        s_pos = np.einsum("bmd,bd->bm", u_pos, v)  # (B, m,)
        s_neg = np.einsum("bkd,bd->bk", u_neg, v)  # (B, k,)

        scale = 1 / (len(context_words[0]) + len(negative_words[0])) / len(center_word)

        proba_pos = sigmoid(s_pos)  # (B, m,)
        proba_neg = sigmoid(s_neg)  # (B, k,)

        pos_loss = -np.log(proba_pos + self.eps).sum()
        neg_loss = -np.log(1.0 - proba_neg + self.eps).sum()

        loss = (pos_loss + neg_loss) * scale

        cache = {
            "v": v, "u_pos": u_pos, "u_neg": u_neg,
            "proba_pos": proba_pos, "proba_neg": proba_neg,
            "context_words": context_words, "negative_words": negative_words,
            "center_word": center_word,
            "scale": scale
        }
        return loss, cache

    def backward(self, cache: dict):
        v = cache["v"]  # (B, D,)
        u_pos = cache["u_pos"]  # (B, m, D)
        u_neg = cache["u_neg"]  # (B, k, D)
        proba_pos = cache["proba_pos"]  # (B, m,)
        proba_neg = cache["proba_neg"]  # (B, k,)
        context_words = cache["context_words"]
        negative_words = cache["negative_words"]
        center_word = cache["center_word"]
        scale = cache["scale"]

        g_pos = (proba_pos - 1.0) * scale  # (B, m,)
        g_neg = proba_neg * scale  # (B, k,)

        grad_v = (u_pos * g_pos[:, :, None]).sum(axis=1) + (u_neg * g_neg[:, :, None]).sum(axis=1)

        # we add a dimension, so that we have (B x N x 1) x (1 x B x D) = (B x N x D)
        grad_u_pos = g_pos[:, :, None] * v[:, None, :]  # (B, m, D)
        grad_u_neg = g_neg[:, :, None] * v[:, None, :]  # (B, k, D)

        self.w_in.backward(center_word, grad_v, lr=self.lr)
        self.w_out.backward(context_words, grad_u_pos, lr=self.lr)
        self.w_out.backward(negative_words, grad_u_neg, lr=self.lr)

    def predict_next_word(self, current_word: np.ndarray, top_k: int = 8):
        # we predict 8 words as the model was trained on 8 context words
        current_embeddings = self.w_in.forward(current_word)
        scores = np.dot(current_embeddings, self.w_out.embeddings.T) / (
                    np.linalg.norm(current_embeddings, axis=1, keepdims=True) * np.linalg.norm(self.w_out.embeddings,
                                                                                               axis=1))
        answers = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        return answers
