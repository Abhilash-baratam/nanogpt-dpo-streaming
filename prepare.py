from pathlib import Path
import torch

# 1. Locate and read the dataset
input_path = Path(__file__).resolve().with_name("input.txt")

if not input_path.exists():
    raise FileNotFoundError(f"Cannot find input.txt at {input_path}")

with input_path.open('r', encoding='utf-8') as f:
    text = f.read()

# 2. Vocabulary definitions
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [char_to_id[c] for c in s if c in char_to_id]
decode = lambda l: ''.join([id_to_char[i] for i in l])

# 3. Create PyTorch tensor and split (90% train, 10% val)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# 4. Safe Batch Generator
def get_batch(split, batch_size=32, block_size=32, device='cpu'):
    # Select split
    d = train_data if split == 'train' else val_data
    
    # CRITICAL FIX: If val_data is too short to support block_size, fall back to train_data
    if len(d) <= block_size + 1:
        d = train_data

    # Generate random batch slices
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    print(f"Reading from: {input_path}")
    print(f"Total characters: {len(data):,}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Train size: {len(train_data):,} | Val size: {len(val_data):,}")