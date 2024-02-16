import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor, optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy import mean
import gensim
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.scheme import IOB2
from sklearn.metrics import f1_score as sklearn_f1_score
import sys, getopt
import pickle

from classes import *

def create_item_list(df):
    # Convert to list of dict's,
    item_list = [row.to_dict() for idx, row in df.iterrows()]

    # Shuffle
    np.random.shuffle(item_list)

    # Ensure utt's and iob tags are matched
    item_list = [item for item in item_list if len(item['utterances'].split(' ')) == len(item['IOB Slot tags'])]
    return item_list


def prepare_batch(batch, in_sequencer, bio_sequencer, rel_sequencer, relation_to_idx):
    texts, bio_tags, rel_tags = zip(*batch)
    text_tensor, lengths = in_sequencer.create_padded_tensor(texts)
    bio_tensor, _ = bio_sequencer.create_padded_tensor(bio_tags)
    rel_tensor = rel_sequencer.create_padded_tensor(rel_tags, relation_to_idx)  # TODO: make a part of sequencer
    return text_tensor, lengths, bio_tensor, rel_tensor


def train_loop(model, opt, loss_fn_bio, loss_fn_rel, dataloader, device):
    # Help from https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

    model.train()
    total_loss = 0

    for i, batch in enumerate(dataloader):
        X = batch[0].to(device)
        y_bio = batch[2].to(device)
        y_rel = batch[3].to(device)

        pred_bio, pred_rel = model(X)

        # Permute bio to (batch size, labels, seq_len)
        pred_bio = pred_bio.permute(1, 2, 0)

        # print(pred_bio[0][0])
        # print(f'y_bio shape: {y_bio.shape}')
        # print(f'pred_bio shape: {pred_bio.shape}')
        # print(f'\ny_rel shape: {y_rel.shape}')
        # print(f'pred_rel shape: {pred_rel.shape}')

        loss = loss_fn_rel(pred_rel, y_rel) + loss_fn_bio(pred_bio, y_bio)
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()
        # print(f'{i} / {len(dataloader)}: {loss}')

    return total_loss / len(dataloader)


def val_loop(model, loss_fn_bio, loss_fn_rel, dataloader, bio_sequencer, device):
    model.eval()
    total_loss = 0
    thresh = 0.4

    bio_targets = []
    predicted_bio_labels = []
    rel_targets = []
    predicted_rel_labels = []

    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(device)
            y_bio = batch[2].to(device)
            y_rel = batch[3].to(device)

            logits_bio, logits_rel = model(X)

            # Permute bio to (batch size, labels, seq len)  # TODO: put in model
            logits_bio = logits_bio.permute(1, 2, 0)

            loss = loss_fn_rel(logits_rel, y_rel) + loss_fn_bio(logits_bio, y_bio)
            total_loss += loss.detach().item()

            # Softmax and argmax over possible bio labels
            probs_bio = F.softmax(logits_bio, dim=-2)
            preds_bio = torch.argmax(probs_bio, dim=-2)

            probs_rel = F.sigmoid(logits_rel)  # , dim=1)
            preds_rel = [[int(r > thresh) for r in b] for b in probs_rel.tolist()]

            bio_targets.append(y_bio.tolist())
            predicted_bio_labels.append(preds_bio.tolist())
            rel_targets = rel_targets + y_rel.tolist()
            predicted_rel_labels = predicted_rel_labels + preds_rel

    # Get bio targets and predictions without the padding
    non_padding_bio_targets = [[_x[i] for i in range(len(_x)) if _x[i] != bio_sequencer.pad_index] for _batch in
                               bio_targets for _x in _batch]
    non_padding_bio_predictions = [[_x[i] for i in range(len(_x)) if _x[i] != bio_sequencer.pad_index] for _batch in
                                   predicted_bio_labels for _x in _batch]

    # TODO: hacky, get padding working with model (it's currently predicting padding)
    non_padding_bio_predictions = [[preds[i] for i in range(len(targets))] for preds, targets in
                                   zip(non_padding_bio_predictions, non_padding_bio_targets)]
    # [print(t, p) for t, p in zip(non_padding_bio_targets, non_padding_bio_predictions)]

    # Convert from one hot to BIO text labels
    bio_true_labels = [[str(bio_sequencer.idx2word[_y]).replace('_', '-') for _y in _x] for _x in
                       non_padding_bio_targets]
    bio_predicted_labels = [[str(bio_sequencer.idx2word[_y]).replace('_', '-') for _y in _x][:len(_t)] for _x, _t in
                            zip(non_padding_bio_predictions, bio_true_labels)]

    # Calculate BIO f1
    # [print(t, p) for t, p in zip(bio_true_labels, bio_predicted_labels)]
    bio_f1 = seqeval_f1_score(bio_true_labels, bio_predicted_labels, scheme=IOB2)
    # print(f'BIO tag f1: {bio_f1}')

    # Calculate relation f1
    rel_targets = [[int(t) for t in targets] for targets in rel_targets]
    rel_f1 = sklearn_f1_score(rel_targets, predicted_rel_labels, average=None, zero_division=0)
    # print(f'Relation f1: {mean(rel_f1)}')

    return total_loss / len(dataloader), mean(bio_f1), mean(rel_f1)


def weight_reset(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def train_model(model, opt, loss_fn_bio, loss_fn_rel, train_loader, val_loader, epochs, bio_sequencer, device):
    train_loss_list, val_loss_list, val_bio_f1_list, val_rel_f1_list = [], [], [], []
    print("Training model...")

    for epoch in range(epochs):
        print(f'-----Epoch {epoch}-----')
        train_loss = train_loop(model, opt, loss_fn_bio, loss_fn_rel, train_loader, device)
        train_loss_list.append(train_loss)

        val_loss, bio_f1, rel_f1 = val_loop(model, loss_fn_bio, loss_fn_rel, val_loader, bio_sequencer, device)
        val_loss_list.append(val_loss)
        val_bio_f1_list.append(bio_f1)
        val_rel_f1_list.append(rel_f1)

        print(f"Train loss:      {train_loss}")
        print(f"Validation loss: {val_loss}")
        print(f"BIO f1:          {bio_f1}")
        print(f"Relation f1:     {rel_f1}\n")

    return train_loss_list, val_loss_list, val_bio_f1_list, val_rel_f1_list



def test_model(model, df, device, text_sequencer, bio_sequencer, rel_sequencer, idx_to_relation, output_path):
    model.eval()
    thresh = 0.4

    utterances = df.utterances.tolist()
    print(utterances[-2])
    utts_encoded = [text_sequencer.encode(utt) for utt in utterances]
    print(utts_encoded[-2])

    predicted_bio_labels = []
    predicted_rel_labels = []

    with torch.no_grad():
        for utt in utts_encoded:
            x = torch.tensor(utt)
            x = torch.unsqueeze(x, dim=0).to(device)

            logits_bio, logits_rel = model(x)

            # Permute bio to (batch size, labels, seq len)
            logits_bio = logits_bio.permute(1, 2, 0)

            # Softmax and argmax over possible bio labels
            probs_bio = F.softmax(logits_bio, dim=-2)
            preds_bio = torch.argmax(probs_bio, dim=-2)

            probs_rel = F.sigmoid(logits_rel)  # , dim=1)
            preds_rel = [[int(r > thresh) for r in b] for b in probs_rel.tolist()]

            predicted_bio_labels = predicted_bio_labels + preds_bio.tolist()
            predicted_rel_labels = predicted_rel_labels + preds_rel

    # Convert from one hot to BIO text labels
    predicted_bio_labels = [[bio_sequencer.idx2word[_y] for _y in _x] for _x in predicted_bio_labels]

    # Convert from one hot to relation labels
    predicted_rel_labels = [[idx_to_relation[i] if x==1 else None] for i, x in enumerate(predicted_rel_labels)]

    # Output to csv
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['utterances', 'IOB Slot tags', 'Core Relations'])
        for i, (utt, bio, rel) in enumerate(zip(utterances, predicted_bio_labels, predicted_rel_labels)):
            bio_str = ' '.join(str(x) for x in bio)
            rel_str = ' '.join(str(x) for x in rel)
            writer.writerow([utt, bio_str, rel_str])

    return



def main():
    # Get arguments
    long_options = ["train", "test", "data=", "model_path=", "output=", "save_model="]
    argumentList = sys.argv[1:]
    arguments, values = getopt.getopt(argumentList, "", long_options)
    train = False
    test = False
    output_path = "./preds.csv"
    model_path = "./joint_trained_model.csv"
    save_model_path = "./joint_trained_model.csv"
    save_model = False
    data_path = ""
    for arg, val in arguments:
        if arg == "--train":
            train = True
        if arg == "--test":
            test = True
        if arg == "--data":
            data_path = val
        if arg == "--model_path":
            model_path = val
        if arg == "--output":
            output_path = val
        if arg == "--save_model":
            save_model = True
            save_model_path = val
    if not train and not test:
        print("Please specify train or test.")
        exit()
    if not train != test:
        print('Please only train or test.')
        exit()
    if data_path == "":
        print("Please enter data path.")
        exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if train:
        # Load train csv
        train_csv_df = pd.read_csv(data_path)
        train_csv_df['IOB Slot tags'] = train_csv_df['IOB Slot tags'].replace(np.nan, "none")
        train_csv_df['Core Relations'] = train_csv_df['Core Relations'].replace(np.nan, "none")

        # Tokenize train
        train_csv_df["IOB Slot tags"] = train_csv_df["IOB Slot tags"].apply(lambda tags: tags.split())
        train_csv_df["Core Relations"] = train_csv_df["Core Relations"].apply(lambda tags: tags.split())

        item_list = create_item_list(train_csv_df)
        relation_to_idx = list(sorted(set([j for i in item_list for j in i["Core Relations"]])))
        relation_to_idx = {r: i for i, r in enumerate(relation_to_idx)}
        idx_to_relation = {i: r for i, r in enumerate(relation_to_idx)}

        # Split train csv into train and val
        train_df, val_df = train_test_split(item_list, test_size=0.1, train_size=0.9)

        # Convert to lists
        train_data = [(i['utterances'], i['IOB Slot tags'], i['Core Relations']) for i in train_df]
        val_data = [(i['utterances'], i['IOB Slot tags'], i['Core Relations']) for i in val_df]

        train_texts = [p[0] for p in train_data]
        train_bios = [p[1] for p in train_data]
        train_rels = [p[2] for p in train_data]

        # Load word2vec weights
        try:
            print("Loading embeddings...")
            word2vec_weights = gensim.models.KeyedVectors.load_word2vec_format('word2vec/model.txt')
            print("Loaded embeddings")
        except:
            print("Failed to load embeddings")
            exit()

        text_sequencer = VocabularyEmbedding(word2vec_weights)
        bio_sequencer = BIOTagSequencer(train_bios)
        rel_sequencer = RelationSequencer(train_rels)

        train_dataset = BIORelDataset(train_data, text_sequencer, bio_sequencer, rel_sequencer)
        val_dataset = BIORelDataset(val_data, text_sequencer, bio_sequencer, rel_sequencer)

        train_loader = DataLoader(train_dataset, batch_size=32,
                                  collate_fn=lambda batch:
                                  prepare_batch(batch, text_sequencer, bio_sequencer, rel_sequencer, relation_to_idx))
        val_loader = DataLoader(val_dataset, batch_size=32,
                                collate_fn=lambda batch:
                                prepare_batch(batch, text_sequencer, bio_sequencer, rel_sequencer, relation_to_idx),
                                shuffle=False)

        lr = 1e-3

        model = BIORelationTransformer(num_tokens=len(text_sequencer.w2v),
                                       dim_model=64,
                                       num_heads=2,
                                       num_encoders=2,
                                       dropout_p=0.1,
                                       num_bio_tags=len(bio_sequencer.word2idx),
                                       num_rel_tags=len(relation_to_idx)).to(device)
        opt = optim.Adam(model.parameters(), lr=lr)
        loss_fn_bio = nn.CrossEntropyLoss(ignore_index=bio_sequencer.pad_index)
        loss_fn_rel = nn.BCEWithLogitsLoss()

        epochs = 2  # TODO
        weight_reset(model)
        train_loss_l, val_loss_l, bio_f1_l, rel_f1_l = train_model(model, opt,
                                                                   loss_fn_bio, loss_fn_rel,
                                                                   train_loader, val_loader,
                                                                   epochs, bio_sequencer, device)

        if save_model:
            torch.save(model.state_dict(), save_model_path)

        # Hacky way I get testing working
        sequencers = [text_sequencer, bio_sequencer, rel_sequencer, relation_to_idx]

        text_seq_file = open('text_seq.pickle', 'ab')
        pickle.dump(text_sequencer, text_seq_file)
        bio_seq_file = open('bio_seq.pickle', 'ab')
        pickle.dump(bio_sequencer, bio_seq_file)
        rel_seq_file = open('rel_seq.pickle', 'ab')
        pickle.dump(rel_sequencer, rel_seq_file)
        rel_idx_file = open('rel_idx.pickle', 'ab')
        pickle.dump(relation_to_idx, rel_idx_file)
        idx_rel_file = open('idx_rel.pickle', 'ab')
        pickle.dump(idx_to_relation, idx_rel_file)

    if test:
        test_csv_df = pd.read_csv(data_path)
        text_seq_file = open('text_seq.pickle', 'rb')
        text_sequencer = pickle.load(text_seq_file)
        bio_seq_file = open('bio_seq.pickle', 'rb')
        bio_sequencer = pickle.load(bio_seq_file)
        rel_seq_file = open('rel_seq.pickle', 'rb')
        rel_sequencer = pickle.load(rel_seq_file)
        rel_idx_file = open('rel_idx.pickle', 'rb')
        relation_to_idx = pickle.load(rel_idx_file)
        idx_rel_file = open('idx_rel.pickle', 'rb')
        idx_to_relation = pickle.load(idx_rel_file)
        model = BIORelationTransformer(num_tokens=len(text_sequencer.w2v),
                                       dim_model=64,
                                       num_heads=2,
                                       num_encoders=2,
                                       dropout_p=0.1,
                                       num_bio_tags=len(bio_sequencer.word2idx),
                                       num_rel_tags=len(relation_to_idx)).to(device)
        model.load_state_dict(torch.load(model_path))
        test_model(model, test_csv_df, device, text_sequencer, bio_sequencer, rel_sequencer, idx_to_relation, output_path)


if __name__ == "__main__":
    main()
