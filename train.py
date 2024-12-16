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
# decode = lambda l: ''.join([itos[i] for i in l])
decode = lambda l: ''.join([itos[i] for i in l])

print(encode("hii there!"))
print(decode([46, 47, 47, 1, 58, 46, 43, 56, 43, 2]))