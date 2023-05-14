# Think Rationally about What You See: Continuous Rationale Extraction for Relation Extraction

## PyTorch
The code is based on PyTorch 1.13+. You can find tutorials [here](https://pytorch.org/tutorials/).

## Data
Download the SemEval dataset and place it under the ```semeval``` directory. The data directory should have the following structure:
```
    semeval/
        train_sentence.json
        train_label_id.json
        test_sentence.json
        test_label_id.json
    utils.py
    train.py
    models.py
    dataset.py
```

## Usage
Run the full model on SemEval dataset with default hyperparameters
```
python3 train.py
```
This will train and evaluate our model. You may download the checkpoint from [Google Drive](https://drive.google.com/drive/folders/1GLMqBF2tV7yFg2PwS_a9pmzVb87cgFAA?usp=share_link) as the parameters are tuned based on this checkpoint.

## Acknowledgements
We thank the authors of the original SemEval dataset and the Transformers library for their contributions.
