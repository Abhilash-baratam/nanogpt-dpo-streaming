from pathlib import Path
import torch
import torch.nn.functional as F
from model import NanoGPT
from prepare import vocab_size, encode, decode

device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 32
n_embd = 64
n_head = 4
n_layer = 3

@torch.no_grad()
def generate_controlled(model, prompt_text, max_new_tokens=40, temperature=0.7, top_k=5):
    model.eval()
    tokens = encode(prompt_text)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        
        # Scale logits with temperature
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        # Top-k pruning
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].tolist())


def load_model(filename):
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {filename}. Please check your files.")
    model = NanoGPT(vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

base_model = load_model("nanogpt_base.pt")
dpo_model = load_model("nanogpt_dpo_aligned.pt")

print("Both models loaded onto", device)
print("=" * 65)

test_prompts = [
    "\nROMEO:\nWhat light ",
    "\nKING:\nSpeak, ",
    "\nJULIET:\nO Romeo, "
]

for prompt in test_prompts:
    print(f"\nPrompt: {repr(prompt)}")
    print("-" * 65)
    
    base_out = generate_controlled(base_model, prompt, max_new_tokens=30, temperature=0.7, top_k=5)
    dpo_out = generate_controlled(dpo_model, prompt, max_new_tokens=30, temperature=0.7, top_k=5)

    print(f"[Base Model Output]:\n{base_out}")
    print(f"\n[DPO Aligned Output]:\n{dpo_out}")
    print("=" * 65)