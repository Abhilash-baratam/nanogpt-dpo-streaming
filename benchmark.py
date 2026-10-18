import math
from pathlib import Path
import torch
import torch.nn.functional as F
from model import NanoGPT
from prepare import val_data, vocab_size, encode
from dpo_data import preference_data

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 3
BETA = 0.2


# -----------------------------------------------------------------------------
# 1. Model Loader Helper
# -----------------------------------------------------------------------------
def load_checkpoint(path_str: str) -> NanoGPT:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path_str}. Run your training scripts first.")
    
    model = NanoGPT(vocab_size, n_embd=N_EMBD, block_size=BLOCK_SIZE, n_head=N_HEAD, n_layer=N_LAYER).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# -----------------------------------------------------------------------------
# 2. Perplexity Calculation Function
# -----------------------------------------------------------------------------
@torch.no_grad()
def calculate_perplexity(model: NanoGPT, data: torch.Tensor, max_batches: int = 50, batch_size: int = 32) -> float:
    """
    Computes cross-entropy loss over random slices of validation data
    and returns Perplexity = exp(loss).
    """
    total_loss = 0.0
    model.eval()

    for _ in range(max_batches):
        ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
        x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix]).to(DEVICE)
        y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix]).to(DEVICE)

        _, loss = model(x, y)
        total_loss += loss.item()

    avg_loss = total_loss / max_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


# -----------------------------------------------------------------------------
# 3. DPO Reward Margin Calculation
# -----------------------------------------------------------------------------
@torch.no_grad()
def get_sequence_log_prob(model: NanoGPT, full_ids: torch.Tensor, prompt_len: int) -> float:
    logits, _ = model(full_ids)
    logits = logits[:, :-1, :]
    targets = full_ids[:, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    gathered = torch.gather(log_probs, 2, targets.unsqueeze(-1)).squeeze(-1)

    # Score only the completion tokens
    completion_mask = torch.zeros_like(gathered)
    completion_mask[:, prompt_len - 1:] = 1.0

    return (gathered * completion_mask).sum().item()


@torch.no_grad()
def evaluate_reward_margins(base_model: NanoGPT, aligned_model: NanoGPT):
    """
    Computes implicit rewards r(x, y) = beta * [log pi(y|x) - log ref(y|x)]
    and checks if chosen completions score higher than rejected ones.
    """
    margins = []
    print("\n--- Direct Preference Optimization (DPO) Reward Margins ---")
    
    for idx, item in enumerate(preference_data):
        prompt_tokens = encode(item["prompt"])
        chosen_tokens = encode(item["chosen"])
        rejected_tokens = encode(item["rejected"])
        p_len = len(prompt_tokens)

        # Batch inputs cropped to BLOCK_SIZE
        chosen_seq = torch.tensor([prompt_tokens + chosen_tokens], dtype=torch.long, device=DEVICE)[:, :BLOCK_SIZE]
        rejected_seq = torch.tensor([prompt_tokens + rejected_tokens], dtype=torch.long, device=DEVICE)[:, :BLOCK_SIZE]

        # Calculate log-probabilities
        pi_w = get_sequence_log_prob(aligned_model, chosen_seq, p_len)
        ref_w = get_sequence_log_prob(base_model, chosen_seq, p_len)

        pi_l = get_sequence_log_prob(aligned_model, rejected_seq, p_len)
        ref_l = get_sequence_log_prob(base_model, rejected_seq, p_len)

        # Implicit rewards
        r_chosen = BETA * (pi_w - ref_w)
        r_rejected = BETA * (pi_l - ref_l)
        margin = r_chosen - r_rejected
        margins.append(margin)

        print(f"Sample {idx + 1}: Margin = {margin:+.4f} | r(chosen) = {r_chosen:+.4f} vs r(rejected) = {r_rejected:+.4f}")

    avg_margin = sum(margins) / len(margins)
    print(f"\nAverage Reward Margin: {avg_margin:+.4f} (Positive is aligned)")


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading models for comparative benchmarking...")
    base_model = load_checkpoint("nanogpt_base.pt")
    aligned_model = load_checkpoint("nanogpt_dpo_aligned.pt")

    print("\n--- Validation Perplexity Comparison ---")
    base_loss, base_ppl = calculate_perplexity(base_model, val_data)
    aligned_loss, aligned_ppl = calculate_perplexity(aligned_model, val_data)

    print(f"Base Model    | Val Loss: {base_loss:.4f} | Perplexity: {base_ppl:.2f}")
    print(f"Aligned Model | Val Loss: {aligned_loss:.4f} | Perplexity: {aligned_ppl:.2f}")

    evaluate_reward_margins(base_model, aligned_model)