from torch.utils.data import Dataset
import numpy as np
#from einops import repeat
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from jiwer import wer
import wandb
from torch.utils.data import DataLoader
import random

# Hyper-parameters
embed_size = 256
hidden_size = 1024
num_layers = 2
num_epochs = 18
batch_size = 64
learning_rate = 0.00075
dropout = 0.0
bidirectional = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "train_wer"},
    "parameters": {
        "embed_size": {"values": [250, 500, 1000, 1500, 2000]},
        "hidden_size": {"values": [250, 500, 1000, 1500, 2000]},
        "num_layers": {"values": [1, 2, 3, 4]},
        "learning_rate": {'max': 0.01, 'min': 0.0001},
        "dropout": {"values": [0.0, 0.2, 0.4, 0.6, 0.8]}
    }
}

#Creates vocab dictionaries for input and target
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

#Loads data from file and tokenizes it
class G2PDataset(Dataset):
    def __init__(self, dir_path):
        self.data = []
        self.input = Dictionary()
        self.target = Dictionary()

        self.load_data(dir_path)
            
    def load_data(self, dir_path):
        tokens = []
        # Loading Train data and adding char to vocab
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
                first_word = list(words[0]) + ['<os>']
                rest = list(['<s>'] + words[1:][0].split(' ')) + ['<os>']
                self.data.append((first_word, rest))
                tokens += first_word 
                
        for char in tokens:
            self.input.add_char(char)
            
        # Loading Val data and adding char to vocab
        with open("/academic/CSCI481/202320/project_data/task3/phonemes.txt", 'r') as f:
            text = f.read()
            lines = text.split('\n')
            self.target.add_char('<pad>')
            self.target.add_char('<os>')
            self.target.add_char('<s>')   
            for line in lines:
                self.target.add_char(line)    
        return self.data

    # Gets the data at the index and return it as tokenized tensors
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
    
    # Pads the sequences in a batch to have equal lengths depending on the max length in the batch
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

        # Pad sequences in a batch to have equal lengths
        padded_inputs = []
        padded_targets = []

        for input, target in zip(inputs, targets):
            padded_input = torch.cat((input, (torch.tensor(self.input.char2idx['<pad>'])).repeat(max_input_length - len(input)).to(device)), dim=0)
            padded_target = torch.cat((target, (torch.tensor(self.target.char2idx['<pad>'])).repeat(max_target_length - len(target)).to(device)), dim=0)
            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)
            #print("padded_input", padded_input)
            #print("padded_target", padded_target)

        return padded_inputs, padded_targets

#Creates the encoder and decoder models
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(input_vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.linear_hidden = nn.Linear(hidden_size*2, hidden_size)# *2 because bidirectional 
        self.linear_cell = nn.Linear(hidden_size*2, hidden_size)# *2 because bidirectional
    
    def forward(self, x):
        embedding = self.dropout(self.embed(x))
        output, (hidden, cell) = self.lstm(embedding)
        if(bidirectional):
            hidden = torch.cat((hidden[0:num_layers], hidden[num_layers:num_layers*2]), dim=2)
            hidden = self.linear_hidden(hidden)
            cell = torch.cat((cell[0:num_layers], cell[num_layers:num_layers*2]), dim=2)
            cell = self.linear_cell(cell)
        return output, hidden, cell
    
class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(target_vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, x, hidden, cell):
        x = x.unsqueeze(0)
        embedding = self.embed(x)
        embedding = self.dropout(embedding)
        output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
#        output = self.norm(output)
        prediction = self.linear(output)
        prediction = prediction.squeeze(0)
        return prediction, hidden, cell

train_set = G2PDataset('/academic/CSCI481/202320/project_data/task3/train.txt')
dev_set = G2PDataset('/academic/CSCI481/202320/project_data/task3/dev.txt')

train_loader = DataLoader(train_set, batch_size, shuffle=True, collate_fn=train_set.collate_fn)

class LSTMModel(nn.Module):
    def __init__(self, train_vocab_size, dev_vocab_size, embed_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.encode = Encoder(train_vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.decode = Decoder(dev_vocab_size, embed_size, hidden_size, num_layers, dropout)

    def forward(self, Grapheme, Phoneme, teacher_forcing_ratio=0.5):
        
        batch_size = Phoneme.shape[1]
        Phoneme_len = Phoneme.shape[0]

        dev_vocab_size = train_set.target.__vocablen__()

        outputs = torch.zeros(Phoneme_len, batch_size, dev_vocab_size).to(device)
        output, hidden, cell = self.encode(Grapheme)
        x = Phoneme[0]

        for i in range(1, Phoneme_len): #0 - Phoneme target len
            output, hidden, cell = self.decode(x, hidden, cell)
            outputs[i] = output

            # output = (N, vocab_len)
            predicted = output.argmax(1)
            
            if random.random() < teacher_forcing_ratio:
                x = Phoneme[i]
            else:
                x = predicted
        return outputs


model = LSTMModel(train_set.input.__vocablen__(), train_set.target.__vocablen__(), embed_size, hidden_size, num_layers)
m = model.to(device)

pad_idx = train_set.input.char2idx['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(model, iterator, optimizer, criterion):
    
    cumulative_loss = 0
    cumulative_wer = 0

    for input, target in tqdm(train_loader):    
        fix_input = torch.stack(input, dim=1)
        fix_target = torch.stack(target,dim=1)

        output = model(fix_input, fix_target) # (char token, batch, vocab)

        pred_word = generate_from_model(output)
        target_word = generate_from_target(fix_target)

#        print("pred_word", pred_word)
#        print("target_word", target_word)

        cumulative_wer += wer(target_word, pred_word)

        output = output[1:].reshape(-1, output.shape[2])
        new_target = fix_target[1:].reshape(-1)

        loss = criterion(output, new_target)
        cumulative_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        wandb.log({"train_loss": loss.item(), "train_wer": error_rate})

    avg_loss = cumulative_loss / len(train_loader)
    avg_wer = cumulative_wer / len(train_loader)
    return avg_loss, avg_wer

def evaluation(model, iterator, criterion):

    cumulative_loss = 0
    cumulative_wer = 0

    with torch.no_grad():

        for input, target in tqdm(train_loader):    
            fix_input = torch.stack(input, dim=1)
            fix_target = torch.stack(target,dim=1)
            
            output = model(fix_input, fix_target, 0) # (char token, batch, vocab)   

            pred_word = generate_from_model(output)
            target_word = generate_from_target(fix_target)

            cumulative_wer += wer(target_word, pred_word)

            output = output[1:].reshape(-1, output.shape[2])
            new_target = fix_target[1:].reshape(-1)

            cumulative_loss += criterion(output, new_target).item()

#            wandb.log({"val_loss": eval_loss.item(), "val_wer": error_rate})

        avg_loss = cumulative_loss / len(train_loader)
        avg_wer = cumulative_wer / len(train_loader)
    return avg_loss, avg_wer


# converts tokenized outputs into words
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
        word.pop(0)

        words.append(' '.join(word))
    return words

# converts tokenized targets back into words
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

def main():
    wandb.init(project="g2p_correct_calculation")

    best_loss = float('inf')

    for epoch in tqdm(range(num_epochs)):

        train_loss, train_wer = train(model, train_loader, optimizer, criterion)
        val_loss, val_wer = evaluation(model, train_loader, criterion)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_wer": train_wer, "val_wer": val_wer})

        if val_loss < best_loss:
            model_name = "final-final-model.pt"
            print("Saving '" + model_name + "'")
            best_loss = val_loss
            torch.save(model.state_dict(), model_name)

        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\t Val. Loss: {val_loss:.3f}')
        print(f'\t Train WER: {train_wer:.3f}')
        print(f'\t Val. WER: {val_wer:.3f}')

#sweep_id = wandb.sweep(sweep_config, project="g2p_correct_calculation")

#wandb.agent(sweep_id, function=main, count=15)

if __name__ == "__main__":
    main()
