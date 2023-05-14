import os
import json
import pdb
import time
import argparse
from collections import Counter

import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from models import REModule
from utils import get_dataset
from dataset import RelationDataset

# set transformer not to connect to the internet
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def score(key, prediction, verbose=True, NO_RELATION=0, silent=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print the aggregate score
    if verbose and silent == False:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    if silent == False:
        print("SET NO_RELATION ID: ", NO_RELATION)
        print("Precision (micro): {:.3%}".format(prec_micro))
        print("   Recall (micro): {:.3%}".format(recall_micro))
        print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro

# register args
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="semeval")
args = parser.parse_args()

def main():
    # torch random seed
    rand_seed = 47
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    # for data loader's random shuffle seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config batch size
    batch_size = 16
    # get number of classes
    if args.dataset == "semeval":
        classes = [i for i in range(19)]
    elif args.dataset == "tacred":
        classes = [i for i in range(42)]
    num_classes = len(set(classes))

    re_module = REModule(device, enc_size=128, num_classes=num_classes, ds=args.dataset)
    i = 0

    if args.dataset not in ["semeval", "tacred", "tacred-revisit", "re-tacred"]:
        raise Exception("dataset must be one of semeval/tacred/tacred-revisit/re-tacred")
    
    train_texts, train_relations = get_dataset("train", ds=args.dataset)
    test_texts, test_relations = get_dataset("test", ds=args.dataset)
    train_dataset = RelationDataset(train_texts, train_relations)
    test_dataset = RelationDataset(test_texts[:1000], test_relations[:1000])
        
    # dataloaders
    train_size = len(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
    test_size = len(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset))
    dataset_size = train_size + test_size        
    
    print(f"Preparing to execute. Number of samples in DS: {dataset_size}")
    print("")
    # train
    print(f"Train DS: {train_size} samples")
    re_module.train()
    # init relation extractor
    optimizer = torch.optim.AdamW(re_module.bert_model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
    # load stored model, if any
    epochs = 2
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
    
    
    re_module.bert_model.train()
    
    # check if model is already trained - model.bin

    if os.path.exists(f"./model_{args.dataset}.bin"):
        re_module.bert_model.load_state_dict(torch.load(f"./model_{args.dataset}.bin"))
        print("Finetuned BertForSequenceClassification Model loaded.")
    else:
        print("No model for BertForSequenceClassification is found.")
    
    if input("Finetune BertForSequenceClassification? [no] / yes: ") == "yes":
        # epochs
        for epoch in range(epochs):
            print(f"Epoch {epoch}:")
            for index, (text, relation) in enumerate(train_loader):
                # train classifier
                print(f"Train: {index * batch_size} / {train_size}")

                batch = list(text)
                tok = re_module.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
                b_input_ids, b_token_type_ids, b_attention_mask = tok["input_ids"], tok["token_type_ids"], tok["attention_mask"]
                b_labels = relation.to(device)

                re_module.bert_model.zero_grad()
                forw = re_module.bert_model(
                    b_input_ids, 
                    token_type_ids=b_token_type_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                loss = forw[0]
                logits = forw[1]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(re_module.bert_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # accuracy
                acc = (torch.argmax(logits, dim=1) == b_labels).sum().item() / len(b_labels)
                print(f"Train: {index * batch_size:8d} / {train_size:8d} - loss: {loss.item():8f}, acc: {acc:5f}", end="\r")

    torch.save(re_module.bert_model.state_dict(), f"./model_{args.dataset}.bin")
    print(f"Model saved as './model_{args.dataset}.bin'")
    print()

    # test
    print(f"Test DS: {test_size} samples")
    re_module.eval()
    re_module.bert_model.eval()
    b_labels_all = []
    preds_original_all = []
    preds_altered_all = []

    for index, (text, relation) in enumerate(test_loader):
        # Train the model on the data
        print(f"Test: {index * batch_size:4d} / {test_size:4d}                                             ")
        text_list = list(text)
        mask, altered_enc = re_module.get_binary_mask(text_list)
        tokenized = re_module.bert_tokenize_only(text)
        masked_tokens = tokenized * mask
        recovered_text = re_module.decode_text(masked_tokens.int().to(device), mask)
        recovered_text = [f"[CLS] {text_.replace('[CLS]', '').replace('[SEP]', '').replace(' [PAD]', '')} [SEP]" for text_ in recovered_text]
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            batch_original_text = list(text)
            batch_altered_text = list(recovered_text)
            tok_original = re_module.tokenizer(batch_original_text, padding=True, truncation=True, return_tensors="pt").to(device)
            tok_altered = re_module.tokenizer(batch_altered_text, padding=True, truncation=True, return_tensors="pt").to(device)

            b_input_ids_original, b_token_type_ids_original, b_attention_mask_original = tok_original["input_ids"], tok_original["token_type_ids"], tok_original["attention_mask"]
            b_input_ids_altered, b_token_type_ids_altered, b_attention_mask_altered = tok_altered["input_ids"], tok_altered["token_type_ids"], tok_altered["attention_mask"]
            b_labels = relation.to(device)

            # pdb.set_trace()
            forw_original = re_module.bert_model(
                b_input_ids_original,
                token_type_ids=b_token_type_ids_original,
                attention_mask=b_attention_mask_original,
                labels=b_labels
            )
            forw_altered = re_module.bert_model(
                b_input_ids_altered,
                token_type_ids=b_token_type_ids_altered,
                attention_mask=b_attention_mask_altered,
                labels=b_labels
            )
            loss_original = forw_original[0]
            loss_altered = forw_altered[0]
            logits_original = forw_original[1]
            logits_altered = forw_altered[1]

            # preds
            preds_original = torch.argmax(logits_original, dim=1).flatten()
            preds_altered = torch.argmax(logits_altered, dim=1).flatten()

            # append to all
            b_labels_all.append(b_labels)
            preds_original_all.append(preds_original)
            preds_altered_all.append(preds_altered)

            prec_micro_original, recall_micro_original, f1_micro_original = score(b_labels, preds_original, silent=True)
            prec_micro_altered, recall_micro_altered, f1_micro_altered = score(b_labels, preds_altered, silent=True)


    print(end="\n")
    # concat on dim 0
    b_labels_all = torch.cat(b_labels_all, dim=0)
    preds_original_all = torch.cat(preds_original_all, dim=0)
    preds_altered_all = torch.cat(preds_altered_all, dim=0)

    prec_micro_original, recall_micro_original, f1_micro_original = score(b_labels_all, preds_original_all)
    prec_micro_altered, recall_micro_altered, f1_micro_altered = score(b_labels_all, preds_altered_all)

    print(f"- Precision {prec_micro_original}, Recall {recall_micro_original}, F1 {f1_micro_original}")
    print(f"+ Precision {prec_micro_altered}, Recall {recall_micro_altered}, F1 {f1_micro_altered}")

if __name__ == "__main__":
    main()
