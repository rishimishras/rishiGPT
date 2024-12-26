# read file
with open('input.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))
# print(text[:1000])

# get all the innique chars from the 1 million shakepeare char
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Character set: ",sorted(list(set(text))))
print("size of char set: ", vocab_size)

# creating a mapping from charactes to integer
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there!"))
print(decode([46, 47, 47, 1, 58, 46, 43, 56, 43, 2]))

# let's now encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this

# splitting the first 90% data into training data and rest 10% into validaiton data
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")