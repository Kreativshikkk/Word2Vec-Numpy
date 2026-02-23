# Word2Vec-Numpy

Classic Word2Vec (SkipGram with negative sampling) train loop implementation using Numpy from scratch.

## Structure

- `dataset.py` - is responsible for dataset uploading from Kaggle, processing it and creating batches for training
- `model.py` - is responsible for the Word2Vec model implementation, including forward and backward pass
- `train.py` - is responsible for the training loop
- `eval.py` - is responsible for the evaluation loop
- `accuracy_plot.png` - accuracy plot for eval loop
- `utils.py` - model loading from checkpoint logic
- `config.yaml` - configuration file for training and evaluation parameters
- `main.py` - reads config file, gives access to it for other files
- `requirements.txt` - list of dependencies for the project

## Python version
Python 3.9 is preferred

## Config content

Contains hyperparameters and paths:

```yaml
# training
num_epochs:
window_size:
num_negatives:
batch_size:
kaggle_path:
embedding_dim:
beta_1:
beta_2:
learning_rate:
eps:

# paths
ckpt_directory:
data_directory:
load_from_ckpt:
ckpt_name:
```

## Commands
1. **setup**: `pip install -r requirements.txt` - installs dependencies
2. **download data**: `python dataset.py` - downloads the data from Kaggle and processes it, saves the processed data in `data_directory`
3. **train**: `python train.py` - trains the model and saves the checkpoint in `ckpt_directory`
4. **eval**: `python eval.py` - evaluates the model and saves the accuracy plot

## Evaluation metric and process

Model predicts top 8 similar words for a given word. Then the intersection of predicted words and actual words is
divided by the total number of context words in batch `batch_size * window_size * 2` to get the accuracy metric

## Model weights
You can access the embeddings, trained on 1 epoch [here](https://drive.google.com/file/d/1AWv8mgTnybI8Zj1_8F03zfPyuNzzoPX6/view?usp=sharing)

