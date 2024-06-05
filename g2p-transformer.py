from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from jiwer import wer
import wandb
from torch.utils.data import DataLoader
import random
import torch.nn.functional as F
import pdb

# Hyper-parameters
embed_size = 256
num_layers = 4
num_epochs = 5
batch_size = 64
learning_rate = 0.0005
dropout = 0.1
n_head = 8
max_len = 50
block_size = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "train_wer"},
    "parameters": {
        "embed_size": {"values": [250, 500, 1000, 1500, 2000]},
        "num_layers": {"values": [3, 4, 5]},
        "learning_rate": {'max': 0.01, 'min': 0.0001},
        "n_head": {"values": [2, 4, 8, 16]},
        "dropout": {"values": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]},
    }
}

class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.idx = 0

    def add_char(self, char):
        if not char in self.char2idx:
            self.char2idx[char] = self.idx
            self.idx2char[self.idx] = char
            self.idx += 1

    def __vocablen__(self):
        return len(self.char2idx)


class G2PDataset(Dataset):
    def __init__(self, dir_path):
        self.data = []
        self.input = Dictionary()
        self.target = Dictionary()

        self.load_data(dir_path)

    def load_data(self, dir_path):
        tokens = []
        with open(dir_path, 'r') as f:
            self.input.add_char('<pad>')
            self.input.add_char('<os>')
            self.input.add_char('<s>')
            text = f.read()
            lines = text.split('\n')
            for line in lines:
                if line == '':
                    continue
                words = line.split('  ')
                first_word = list(words[0])# + ['<os>']

#                rest = list(words[1:][0].split(' '))
                rest = list(['<s>'] + words[1:][0].split(' ') + ['<os>'])
                self.data.append((first_word, rest))
                tokens += first_word

            #tokenization
            for char in tokens:
                self.input.add_char(char)

            with open("/academic/CSCI481/202320/project_data/task3/phonemes.txt", 'r') as f:
                text = f.read()
                lines = text.split('\n')
                self.target.add_char('<pad>')
                self.target.add_char('<os>')
                self.target.add_char('<s>')
                for line in lines:
                    self.target.add_char(line)
            #print("input", self.input.char2idx)
            #print("target", self.target.char2idx)
        return self.data

    def __getitem__(self, idx):
        text1, text2 = self.data[idx]
        first = []
        rest = []
        for char in text1:
            first.append(self.input.char2idx[char])
        for char in text2:
            rest.append(self.target.char2idx[char])
        return torch.tensor(first, dtype=torch.long).to(device), torch.tensor(rest, dtype=torch.long).to(device)

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        input_lengths = []
        target_lengths = []
        inputs = []
        targets = []

        for item in batch:
            input, target = item
            inputs.append(input)
            targets.append(target)
            input_lengths.append(len(input))
            target_lengths.append(len(target))

        max_input_length = max(input_lengths)
        max_target_length = max(target_lengths)
        max_len = max(max_input_length, max_target_length)
        # Pad sequences in a batch to have equal lengths
        padded_inputs = []
        padded_targets = []
        
        for input, target in zip(inputs, targets):
            padded_input = torch.cat((input, (torch.tensor(self.input.char2idx['<pad>'])).repeat(max_len - len(input)).to(device)), dim=0)
            padded_target = torch.cat((target, (torch.tensor(self.target.char2idx['<pad>'])).repeat(max_len - len(target)).to(device)), dim=0)
            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)
            #print("padded_input", padded_input)
            #print("padded_target", padded_target)

        return padded_inputs, padded_targets


def generate_from_model(model):
    words = []
    for batch in range(0, model.size(1)):
        word = []
        for char in range(0, model.size(0)):
            num = (model[char][batch].argmax(dim=0).item())
            guess = (train_set.target.idx2char[num])
            if guess == '<os>':
                break
            word.append(guess)
        if len(word) > 1:
            word.pop(0)

        words.append(' '.join(word))
    return words

def generate_from_target(target):
    words = []
    for batch in range(0, target.size(1)):
        word = []
        for char in range(0, target.size(0)):
            num = target[char][batch].item()
            guess = (train_set.target.idx2char[num])
            if guess == '<os>':
                break
            word.append(guess)
        word.pop(0)
        words.append(' '.join(word))
    return words                           

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        B,T,C = k.shape
        k = self.key(k)   # (B,T,C)
        q = self.query(q) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        if mask is not None:
            wei = wei.masked_fill(mask == 0, float('-inf')) # (B, T, T)
        #wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(v) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        out = torch.cat([h(q, k, v, mask) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),  
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PosEncoding(nn.Module):
    def __init__(self, max_len, embed_size, device):
        super().__init__()
        self.embed_size = embed_size
        pos_enc = torch.zeros(max_len, embed_size)   # Sin/Cos positional embeddings
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2, device=device).float() * (-np.log(10000.0) / embed_size)) 
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0) # (max_len, 1, embed_size)
        self.register_buffer('pos_enc', pos_enc)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x * np.sqrt(self.embed_size)
        seq_len = x.size(1)
        x = x + self.pos_enc[:,:seq_len,:]
        return x

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, dropout=dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        attention = self.sa(q, k, v, mask) # attention
        x = self.dropout(self.ln(q + attention)) # residual connection and layer norm
        x = self.dropout(self.ln(x + self.ffwd(x))) # adds the output of feedforward and residual connection
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, head_size, dropout=dropout):
        super().__init__()
        head_size = embed_size // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.block = Block(embed_size, n_head)
        self.ffwd = FeedFoward(embed_size)
        self.ln = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        attention = self.sa(q, q, q, mask) # attention
        query = self.dropout(self.ln(q + attention)) # residual connection and layer norm
        out = self.block(query, k, v, mask) # adds the output of feedforward and residual connection
        return out 
    
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_vocab_size, embed_size)
#        self.pos_enc = nn.Embedding(50, embed_size)
        self.pos_enc = PosEncoding(max_len, embed_size, device)
        self.layers = nn.ModuleList([Block(embed_size, num_heads) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(device) # sequential numbers up to sequence length repeated N times
        x = self.embed(x)
        out = self.pos_enc(x)
        out = self.dropout(out)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.embed = nn.Embedding(target_vocab_size, embed_size)
#        self.pos_enc = nn.Embedding(50, embed_size)
        self.pos_enc = PosEncoding(max_len, embed_size, device)
        self.layers = nn.ModuleList([DecoderBlock(embed_size, num_heads, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, mask):
        N, seq_len = x.shape
        pos = torch.arange(0, seq_len).expand(N, seq_len).to(device) # sequential numbers up to sequence length repeated N times
        x = self.embed(x)
        out = self.pos_enc(x)
        out = self.dropout(out)
        
        for layer in self.layers:
            out = layer(out, enc_out, enc_out, mask)
        return out

class Transformer(nn.Module):
    def __init__(self, input_vocab_size, target_vocab_size, embed_size, num_heads, num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(input_vocab_size, embed_size, num_layers, num_heads, dropout)
        self.decoder = Decoder(target_vocab_size, embed_size, num_layers, num_heads, dropout)
        self.linear = nn.Linear(target_vocab_size, embed_size)
        self.out = nn.Linear(embed_size, target_vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.pad_idx_input = train_set.input.char2idx['<pad>']
        self.pad_idx_target = train_set.target.char2idx['<pad>']

    def gen_mask(self, x):
        N, seq_len = x.shape
        mask = torch.tril(torch.ones((seq_len, seq_len))).expand(N, seq_len, seq_len).to(device)
        return mask


    def forward(self, src, trg):
        trg_mask = self.gen_mask(trg)
        enc_src = self.encoder(src, mask=None)
        out = self.out(self.decoder(trg, enc_src, trg_mask))
        out = self.softmax(out)
        return out


train_set = G2PDataset('/academic/CSCI481/202320/project_data/task3/train.txt')
dev_set = G2PDataset('/academic/CSCI481/202320/project_data/task3/dev.txt')
train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=train_set.collate_fn)


model = Transformer(train_set.input.__vocablen__(), train_set.target.__vocablen__(), embed_size, n_head, num_layers, dropout).to(device)
m = model.to(device)


pad_idx = train_set.input.char2idx['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
cum_loss = 0

def train(model, iterator, optimizer, criterion):
    print("train")
    cum_loss = 0
    cum_error_rate = 0
    for input, target in tqdm(train_loader):
        fix_input = torch.stack(input, dim=0)
        fix_target = torch.stack(target,dim=0)
        output = model(fix_input, fix_target).permute(1,0,2)
        

        fix_target = fix_target.T
        pred_word = generate_from_model(output)
        target_word = generate_from_target(fix_target)
        
#        print("pred_word: ", pred_word)
#        print("target_word: ", target_word)

        error_rate = wer(target_word, pred_word)
        cum_error_rate += error_rate
        
        output = output[1:].reshape(-1, output.shape[2])
        new_target = fix_target[1:].reshape(-1)
        
        loss = criterion(output, new_target)
        cum_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    avg_loss = cum_loss / len(train_loader)
    avg_error_rate = cum_error_rate / len(train_loader)
    return avg_loss, avg_error_rate   

def evaluation(model, iterator, criterion):
    print("eval")
    cum_loss = 0
    cum_error_rate = 0
    for input, target in tqdm(train_loader):
        fix_input = torch.stack(input, dim=0)
        fix_target = torch.stack(target,dim=0)
        output = model(fix_input, fix_target).permute(1,0,2)

        fix_target = fix_target.T
        pred_word = generate_from_model(output)
        target_word = generate_from_target(fix_target)

#        print("pred_word: ", pred_word)
#        print("target_word: ", target_word)

        
        error_rate = wer(target_word, pred_word)
        cum_error_rate += error_rate
        
        output = output[1:].reshape(-1, output.shape[2])
        new_target = fix_target[1:].reshape(-1)
        
        loss = criterion(output, new_target)
        cum_loss += loss.item()

    avg_loss = cum_loss / len(train_loader)
    avg_error_rate = cum_error_rate / len(train_loader)
    return avg_loss, avg_error_rate   

def main():
    wandb.init(project="g2p_transformer")
    best_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):
        train_loss, train_wer = train(model, train_loader, optimizer, criterion)
        val_loss, val_wer = evaluation(model, train_loader, criterion)
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_wer": train_wer, "val_wer": val_wer})

        if val_loss < best_loss:
            model_name = "low-wer-transformer-best-model.pt"
            print("Saving '" + model_name + "'")
            best_loss = val_loss
            torch.save(model.state_dict(), model_name)
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f}')
        print(f'\t Train WER: {train_wer:.3f}')
        print(f'\t Val. WER: {val_wer:.3f}')

#sweep_id = wandb.sweep(sweep_config, project="g2p_transformer")
#wandb.agent(sweep_id, function=main, count=15)
if __name__ == "__main__":
    main()  
