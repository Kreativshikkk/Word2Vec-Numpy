from __future__ import annotations

import pickle
from pathlib import Path
import shutil
import kagglehub
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from hallucination_detection.main import CONFIG

DATA_DIRECTORY = Path(CONFIG.get("data_directory", Path("data")))
KAGGLE_PATH = CONFIG.get("kaggle_path", "kritanjalijain/amazon-reviews")

if KAGGLE_PATH == "":
    raise ValueError("kaggle_path cannot be an empty string")

BATCH_SIZE = CONFIG.get("batch_size", 32)
NUM_NEGATIVES = CONFIG.get("num_negatives", 15)
WINDOW_SIZE = CONFIG.get("window_size", 4)

files = ["test.csv", "train.csv"]
directory_data = Path(__file__).parent / DATA_DIRECTORY
directory_data.mkdir(parents=True, exist_ok=True)

dataset_dir = Path(kagglehub.dataset_download(KAGGLE_PATH))


def save_dict_pickle(obj: dict, path: str | Path) -> None:
    path = Path(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_dict_pickle(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def index_words(text):
    word_2_id, id_2_freqs, id_2_word = {}, {}, {}
    texts = list(text)

    for t in tqdm(texts):
        if not isinstance(t, str):
            continue
        for w in t.split():
            if w in word_2_id:
                wid = word_2_id[w]
                id_2_freqs[wid] = id_2_freqs.get(wid, 0) + 1
            else:
                wid = len(word_2_id)
                id_2_word[wid] = w
                word_2_id[w] = wid
                id_2_freqs[wid] = 1

    save_dict_pickle(word_2_id, directory_data / "word2id.pkl")
    save_dict_pickle(id_2_freqs, directory_data / "id2freqs.pkl")
    save_dict_pickle(id_2_word, directory_data / "id2word.pkl")


def preprocess_file(file_path):
    _df = pd.read_csv(file_path)
    _df.drop(_df.columns[0:2], axis=1, inplace=True)
    _df.dropna(inplace=True)
    return _df[_df.columns[0]]


class CustomDataset:
    def __init__(self, df_path, window_size=WINDOW_SIZE):
        self.texts = preprocess_file(df_path)
        self.window_size = window_size
        self.word2id = load_dict_pickle(directory_data / "word2id.pkl")
        self.id2freqs = load_dict_pickle(directory_data / "id2freqs.pkl")
        self.total_tokens = sum(self.id2freqs.values())

        self.p_keep = np.ones(len(self.word2id), dtype=np.float32)
        for wid, freq in self.id2freqs.items():
            freq /= self.total_tokens
            self.p_keep[wid] = min(1.0, np.sqrt(2e-2 / freq) + 2e-2 / freq) if freq > 0 else 1.0  # keep almost always
        self.random = np.random.default_rng()

    def __len__(self):
        return len(self.texts)

    def sample_frequent_word(self, wid_sequence):
        wid_sequence = np.array(wid_sequence)
        probas = self.p_keep[wid_sequence]
        mask = self.random.random(size=probas.shape) < probas
        return wid_sequence[mask]

    def get_word(self, file_index, word_index):
        text = self.texts[file_index]
        words = text.split()
        if word_index < len(words):
            try:
                return self.word2id[words[word_index]]
            except KeyError:
                return None
        else:
            return None

    def get_context_window(self, file_index, word_index):
        text = self.texts[file_index]
        words = text.split()
        start_index = max(0, word_index - self.window_size)
        end_index = min(len(words), word_index + self.window_size + 1)
        needed_words = words[start_index:word_index] + words[word_index + 1:end_index]
        id_sequence = [self.word2id.get(word, -1) for word in needed_words]
        if -1 in id_sequence or len(id_sequence) == 0:
            return None
        return self.sample_frequent_word(id_sequence)

    def get_negatives(self, file_index, word_index, num_negatives=NUM_NEGATIVES):
        target_seq = self.get_context_window(file_index, word_index)
        negatives = []
        while len(negatives) < num_negatives:
            random_word = np.random.randint(low=0, high=len(self.word2id))
            if random_word not in target_seq:
                negatives.append(random_word)
        return self.sample_frequent_word(negatives)


class BatchLoader:
    def __init__(self, dataset: CustomDataset, window_size=WINDOW_SIZE, num_negatives=NUM_NEGATIVES,
                 batch_size=BATCH_SIZE, last_iter=False):
        self.dataset = dataset
        self.context_length = window_size * 2
        self.num_negatives = num_negatives
        self.batch_size = batch_size
        self.iteration = 0
        self.last_iter = last_iter

    def __len__(self):
        return sum(self.dataset.id2freqs.values()) // self.batch_size

    def __iter__(self):
        center_buf = []
        context_buf = []
        negatives_buf = []

        for text_index in range(len(self.dataset)):
            text = self.dataset.texts[text_index]
            words = text.split()
            for word_index in range(len(words)):
                center_word = self.dataset.get_word(text_index, word_index)
                context_window = self.dataset.get_context_window(text_index, word_index)
                if context_window is None or len(context_window) != self.context_length or center_word is None:
                    continue
                negatives = self.dataset.get_negatives(text_index, word_index, self.num_negatives)
                if len(negatives) < self.num_negatives:
                    continue
                center_buf.append(center_word)
                context_buf.append(context_window)
                negatives_buf.append(negatives)
                if len(center_buf) >= self.batch_size:
                    self.iteration += 1
                    if self.last_iter and self.iteration < self.last_iter:
                        center_buf, context_buf, negatives_buf = [], [], []
                        continue
                    yield center_buf, context_buf, negatives_buf
                    center_buf, context_buf, negatives_buf = [], [], []


if __name__ == "__main__":
    for name in files:
        src = dataset_dir / name
        dst = directory_data / name
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")

    df = preprocess_file(directory_data / "train.csv")  # series
    index_words(df)
