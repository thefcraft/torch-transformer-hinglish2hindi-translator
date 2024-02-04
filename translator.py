import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# PAD_START = ['*']
# PAD_NULL = [' ']
# TOKEN_UNK
TOKEN_PAD = ' '
TOKEN_SOS = '*'
TOKEN_EOS = '^'

chr_eng = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
chr_hin = ['ँ', 'ं', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ऑ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', '़', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॅ', 'े', 'ै', 'ॉ', 'ो', 'ौ', '्', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', '॥']

eng_vocab = [TOKEN_PAD,TOKEN_SOS,TOKEN_EOS]+chr_eng
hin_vocab = [TOKEN_PAD,TOKEN_SOS,TOKEN_EOS]+chr_hin


def add_startToken(texts, start_token='*'):
    return [start_token + text for text in texts]
def add_endToken(texts, end_token='^'):
    return [text + end_token for text in texts]
def preprocesser(texts: list[list], prePadding=True, vocab=eng_vocab, startToken=False, endToken=False, batch_first=False):
    if startToken:
        texts = add_startToken(texts)
    if endToken:
        texts = add_endToken(texts)
    # Convert characters to integers (ASCII - 97)
    text_ints = [[vocab.index(c) for c in text] for text in texts]
    # Apply pre-padding to each sequence
    if prePadding:
        max_length = max(len(seq) for seq in text_ints)
        padded_seqs = pad_sequence([torch.cat([torch.zeros(max_length - len(seq), dtype=torch.int64), torch.LongTensor(seq)]) for seq in text_ints], batch_first=True)
    else:
        padded_seqs = pad_sequence([torch.LongTensor(seq) for seq in text_ints], batch_first=True, padding_value=0)
    
    return padded_seqs.to(device=device) if batch_first else padded_seqs.T.to(device=device)

def load_checkpoint(filename, model, optimizer=None):
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_len,
        device,
        useOneHotEncoder=False
    ):
        super(Transformer, self).__init__()
        if useOneHotEncoder:
            embedding_size = max([src_vocab_size, max_len, trg_vocab_size])
            embedding_size = max([src_vocab_size, max_len, trg_vocab_size])
            embedding_size += embedding_size%8
            print('Embedding size : ', embedding_size)
            self.dropout = lambda x: x
            self.src_word_embedding = lambda x: F.one_hot(x, embedding_size).float()
            self.src_position_embedding = lambda x: F.one_hot(x, embedding_size).float()
            self.trg_word_embedding = lambda x: F.one_hot(x, embedding_size).float()
            self.trg_position_embedding = lambda x: F.one_hot(x, embedding_size).float()
        else:
            self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
            self.src_position_embedding = nn.Embedding(max_len, embedding_size)
            self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
            self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        src_mask = src.transpose(0, 1) == self.src_pad_idx

        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_length)
            .unsqueeze(1)
            .expand(trg_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_trg = self.dropout(
            (self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))
        )

        src_padding_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(
            self.device
        )

        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=trg_mask,
        )
        out = self.fc_out(out)
        return out

class Hinglish2HindiTranslator:
    def __init__(self, model_path='model/hinglish2hindi_epoch-50.pth.tar', model_type_oneHot=False):
        
        # Model hyperparameters
        src_vocab_size = len(eng_vocab)
        trg_vocab_size = len(hin_vocab)
        embedding_size = 128
        num_heads = 8
        assert embedding_size%num_heads == 0
        num_encoder_layers = 2
        num_decoder_layers = 2
        dropout = 0.10
        max_len = 72 if model_type_oneHot else 50
        forward_expansion = 4
        src_pad_idx = eng_vocab.index(TOKEN_SOS)
        if model_type_oneHot: 
            self.model = Transformer(
                None,
                src_vocab_size,
                trg_vocab_size,
                src_pad_idx,
                num_heads,
                num_encoder_layers,
                num_decoder_layers,
                forward_expansion,
                dropout,
                max_len,
                device,
                useOneHotEncoder=True
            ).to(device)
            load_checkpoint(filename=f'model\\hinglish2hindi_oneHot_epoch-50.pth.tar', model=self.model, optimizer=None)
        else:
            self.model = Transformer(
                            embedding_size,
                            src_vocab_size,
                            trg_vocab_size,
                            src_pad_idx,
                            num_heads,
                            num_encoder_layers,
                            num_decoder_layers,
                            forward_expansion,
                            dropout,
                            max_len,
                            device).to(device)

            load_checkpoint(filename=f'model\\hinglish2hindi_epoch-50.pth.tar', model=self.model, optimizer=None)
        self.model.eval()

    def translate(self, text, max_length=50):
        result = []
        for word in text.split(' '):
            x = preprocesser([word], startToken=True, vocab=eng_vocab, endToken=True)
            stopIdx = hin_vocab.index(TOKEN_EOS)
            outputs = []
            for i in range(max_length):
                y = preprocesser([''.join(outputs)], startToken=True, vocab=hin_vocab, endToken=False)
                # print(y)
                with torch.no_grad():
                    output = self.model(x, y)
                best_guess = output.argmax(2)[-1, :].item()
                # print(best_guess)
                if best_guess == stopIdx: break
                outputs.append(hin_vocab[best_guess])
            result.append(''.join(outputs))
        return ' '.join(result)
    