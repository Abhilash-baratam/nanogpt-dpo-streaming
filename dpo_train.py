import copy
from pathlib import Path
import torch
import torch.nn.functional as F
from model import NanoGPT
from prepare import vocab_size, encode, decode, train_data
from dpo_data import preference_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 32
n_embd = 64
n_head = 4
n_layer = 3

# Hyperparameters
beta = 0.2
learning_rate = 1e-5     # Slightly lower rate for stable alignment
num_epochs = 6           # Fewer epochs to prevent over-optimization
nll_weight = 0.5         # Regularization weight to preserve perplexity

# 1. Load models
policy_model = NanoGPT(vocab_size, n_embd=n_embd, block_size=block_size, n_head=n_head, n_layer=n_layer).to(device)
base_weights_path = Path("nanogpt_base.pt")
if not base_weights_path.exists():
    raise FileNotFoundError("Run train.py first to produce nanogpt_base.pt.")

policy_model.load_state_dict(torch.load(base_weights_path, map_location=device))
ref_model = copy.deepcopy(policy_model)
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)


# 2. Sequence log-probability extractor
def get_log_probs(model, input_ids, prompt_len):
    logits, _ = model(input_ids)
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

    completion_mask = torch.zeros_like(gathered_log_probs)
    completion_mask[:, prompt_len - 1:] = 1.0

    return (gathered_log_probs * completion_mask).sum(dim=-1)


# 3. Combined DPO + NLL Loss
def compute_regularized_dpo_loss(policy_model, full_chosen, full_rejected, prompt_len, ref_chosen_lp, ref_rejected_lp):
    # Forward pass on active policy
    policy_chosen_lp = get_log_probs(policy_model, full_chosen, prompt_len)
    policy_rejected_lp = get_log_probs(policy_model, full_rejected, prompt_len)

    # DPO Objective
    pi_logratios = policy_chosen_lp - policy_rejected_lp
    ref_logratios = ref_chosen_lp - ref_rejected_lp
    logits = beta * (pi_logratios - ref_logratios)
    dpo_loss = -F.logsigmoid(logits).mean()

    # Auxiliary NLL loss on chosen sequence to prevent catastrophic forgetting
    chosen_logits, _ = policy_model(full_chosen)
    nll_loss = F.cross_entropy(
        chosen_logits[:, :-1, :].reshape(-1, vocab_size),
        full_chosen[:, 1:].reshape(-1)
    )

    total_loss = dpo_loss + (nll_weight * nll_loss)
    return total_loss, dpo_loss.item(), nll_loss.item()


# 4. Training Loop
print("Starting Regularized DPO (Preserving Perplexity)...")
policy_model.train()

for epoch in range(num_epochs):
    total_dpo, total_nll = 0.0, 0.0

    for item in preference_data:
        prompt_tokens = encode(item["prompt"])
        chosen_tokens = encode(item["chosen"])
        rejected_tokens = encode(item["rejected"])
        prompt_len = len(prompt_tokens)

        full_chosen = torch.tensor([prompt_tokens + chosen_tokens], dtype=torch.long, device=device)[:, :block_size]
        full_rejected = torch.tensor([prompt_tokens + rejected_tokens], dtype=torch.long, device=device)[:, :block_size]

        # Reference log-probs (cached/frozen)
        with torch.no_grad():
            ref_chosen_lp = get_log_probs(ref_model, full_chosen, prompt_len)
            ref_rejected_lp = get_log_probs(ref_model, full_rejected, prompt_len)

        loss, d_loss, n_loss = compute_regularized_dpo_loss(
            policy_model, full_chosen, full_rejected, prompt_len, ref_chosen_lp, ref_rejected_lp
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_dpo += d_loss
        total_nll += n_loss

    if epoch % 2 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch:2d}/{num_epochs - 1} | DPO Loss: {total_dpo/len(preference_data):.4f} | NLL Loss: {total_nll/len(preference_data):.4f}")

# Overwrite checkpoint with calibrated weights
torch.save(policy_model.state_dict(), "nanogpt_dpo_aligned.pt")
print("Saved regularized checkpoint to nanogpt_dpo_aligned.pt!")