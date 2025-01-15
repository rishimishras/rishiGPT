import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 32 # number of independent sequnces that will run in parallel
block_size = 8 # size of the batch/sequence
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

# for reproducibility
torch.manual_seed(1337)

# read file
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

# get all the innique chars from the 1 million shakepeare char
chars = sorted(list(set(text)))
vocab_size = len(chars)

# creating a mapping from charactes to integer
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encoding the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# splitting the first 90% data into training data and rest 10% into validaiton data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # this is the embedding table created for the entire vocabulary of size vocab(65 here) by vocab wich is 50k in gpt-3 refer your written docs
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        print("self.token_embedding_table", self.token_embedding_table)

    def forward(self, idx, target=None):
        logits = self.token_embedding_table(idx) #(B,T,C)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)

            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop=
for iter in range(max_iters):
    if iter == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))