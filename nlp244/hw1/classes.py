import math

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset


class VocabularyEmbedding(object):
    # For representing vocab embeddings
    def __init__(self, gensim_w2v):

        self.w2v = gensim_w2v
        self.w2v.add_vector('<s>', np.random.uniform(low=-1, high=1.0, size=(300,)))
        self.w2v.add_vector('</s>', np.random.uniform(low=-1, high=1.0, size=(300,)))
        self.w2v.add_vector('<pad>', np.random.uniform(low=-1, high=1.0, size=(300,)))
        self.w2v.add_vector('<unk>', np.random.uniform(low=-1, high=1.0, size=(300,)))

        bos = self.w2v.key_to_index.get('<s>')
        eos = self.w2v.key_to_index.get('</s>')
        pad = self.w2v.key_to_index.get('<pad>')
        unk = self.w2v.key_to_index.get('<unk>')

        self.bos_index = bos
        self.eos_index = eos
        self.pad_index = pad
        self.unk_index = unk

    def tokenizer(self, text):
        return [t for t in text.split(' ')]

    def encode(self, text):

        sequence = []

        tokens = self.tokenizer(text)
        for token in tokens:
            index = self.w2v.key_to_index.get(token, self.unk_index)
            sequence.append(index)

        return sequence

    def create_padded_tensor(self, sequences):
        # sequences:
        # print(sequences)

        lengths = [len(sequence) for sequence in sequences]
        max_seq_len = max(lengths)
        tensor = torch.full((len(sequences), max_seq_len), self.pad_index, dtype=torch.long)

        for i, sequence in enumerate(sequences):
            for j, token in enumerate(sequence):
                tensor[i][j] = token

        return tensor, lengths


class BIOTagSequencer(object):
    # For representing BIO tags
    def __init__(self, tag_corpus, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>'):
        self.word2idx = {}
        self.idx2word = {}
        self.unk_index = self.add_token(unk_token)
        self.pad_index = self.add_token(pad_token)
        self.bos_index = self.add_token(bos_token)
        self.eos_index = self.add_token(eos_token)

        for _tags in tag_corpus:
            for _token in self.tokenizer(_tags):
                self.add_token(_token)

    @staticmethod
    def tokenizer(text):
        return [t for t in text]

    def add_token(self, token):
        if token not in self.word2idx:
            self.word2idx[token] = new_index = len(self.word2idx)
            self.idx2word[new_index] = token
            return new_index

        else:
            return self.word2idx[token]

    def encode(self, text):
        tokens = self.tokenizer(text)

        sequence = []

        for token in tokens:
            index = self.word2idx.get(token, self.unk_index)
            sequence.append(index)

        return sequence

    def create_padded_tensor(self, sequences):

        lengths = [len(sequence) for sequence in sequences]
        max_seq_len = max(lengths)
        tensor = torch.full((len(sequences), max_seq_len), self.pad_index, dtype=torch.long)

        for i, sequence in enumerate(sequences):
            for j, token in enumerate(sequence):
                tensor[i][j] = token

        return tensor, lengths


class RelationSequencer:
    # For representing relations
    def __init__(self, relations):
        pass

    def encode(self, text):
        return text

    def create_padded_tensor(self, sequences, relation_to_idx):
        tensor = torch.full(size=(len(sequences), len(relation_to_idx)), fill_value=0, dtype=torch.float)
        for i, sequence in enumerate(sequences):
            for token in sequence:
                tensor[i][relation_to_idx[token]] = 1
        return tensor


class PositionalEncoding(nn.Module):
    # Copied directly from PyTorch docs
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# %%
class BIORelDataset(Dataset):
    def __init__(self, data, text_sequencer, bio_sequencer, rel_sequencer):
        self.data = data
        self.input_sequencer = text_sequencer
        self.bio_sequencer = bio_sequencer
        self.rel_sequencer = rel_sequencer

    def __getitem__(self, index):
        text, tags, relations = self.data[index]
        x = self.input_sequencer.encode(text)
        y_bio = self.bio_sequencer.encode(tags)
        y_rel = self.rel_sequencer.encode(relations)
        return x, y_bio, y_rel

    def __len__(self):
        return len(self.data)


class BIORelationTransformer(nn.Module):
    def __init__(self, num_tokens, dim_model, dropout_p, num_heads, num_encoders, num_bio_tags, num_rel_tags):
        super().__init__()

        # Info
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # Layers
        self.positional_encoder = PositionalEncoding(d_model=dim_model, dropout=dropout_p)
        self.embedding = nn.Embedding(num_tokens, dim_model)

        # encoder_layer = nn.TransformerEncoderLayer(dim_model, nhead=num_heads)
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoders,
            num_decoder_layers=num_encoders,  # TODO
            dropout=dropout_p
        )

        self.out_rel = nn.Linear(dim_model, num_rel_tags)
        self.out_bio = nn.Linear(dim_model, num_bio_tags)

    def forward(self, x):
        # Input shape: (batch size, seq len)
        x = self.embedding(x) * math.sqrt(self.dim_model)
        x = self.positional_encoder(x)

        # Reshape to (seq len, batch size, dim model)
        x = x.permute(1, 0, 2)

        # Transformer blocks
        # t_out = self.transformer_encoder(x)
        # mask = self.transformer.generate_square_subsequent_mask(len(x))  # TODO: create padding mask
        t_out = self.transformer(x, x)

        # print(f'transformer out shape: {t_out.shape}')

        out_iob = self.out_bio(t_out)
        out_rel = self.out_rel(torch.max(t_out, dim=0).values)  # Get rid of seq len dim

        # print(f'out_iob shape: {out_iob.shape}')
        # print(f'out_rel shape: {out_rel.shape}')

        return out_iob, out_rel
