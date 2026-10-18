import torch
from prepare import vocab_size, decode, get_batch
from model import NanoGPT

# Hyperparameters
batch_size = 32
block_size = 32
max_iters = 2000
eval_interval = 200
learning_rate = 1e-3
eval_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# Initialize NanoGPT model
model = NanoGPT(vocab_size, n_embd=64, block_size=block_size, n_head=4, n_layer=3, dropout=0.2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Device: {device} | Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("-" * 50)

# Training loop
for iter_num in range(max_iters):
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter_num:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")

    xb, yb = get_batch('train', batch_size, block_size, device)
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate sample text
print("\n" + "=" * 50)
print("Generated Text Sample:")
print("=" * 50)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_tokens = model.generate(context, max_new_tokens=250)[0].tolist()
print(decode(generated_tokens))
# Save pretrained base model weights
torch.save(model.state_dict(), "nanogpt_base.pt")
print("Base model weights saved to nanogpt_base.pt")