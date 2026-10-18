from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from model import NanoGPT
from prepare import vocab_size, encode, decode

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BLOCK_SIZE = 32
N_EMBD = 64
N_HEAD = 4
N_LAYER = 3
CHECKPOINT_PATH = Path("nanogpt_dpo_aligned.pt")

model_instance: Optional[NanoGPT] = None

# -----------------------------------------------------------------------------
# Lifespan: Model Loader
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_instance
    target_path = CHECKPOINT_PATH if CHECKPOINT_PATH.exists() else Path("nanogpt_base.pt")

    if not target_path.exists():
        raise FileNotFoundError("Missing checkpoint! Run train.py or dpo_train.py first.")

    print(f"Loading checkpoint '{target_path}' onto {DEVICE}...")
    model_instance = NanoGPT(
        vocab_size=vocab_size,
        n_embd=N_EMBD,
        block_size=BLOCK_SIZE,
        n_head=N_HEAD,
        n_layer=N_LAYER
    ).to(DEVICE)

    model_instance.load_state_dict(torch.load(target_path, map_location=DEVICE))
    model_instance.eval()
    print("NanoGPT model successfully loaded!")
    yield
    model_instance = None


app = FastAPI(title="NanoGPT Web App", lifespan=lifespan)

# -----------------------------------------------------------------------------
# Schemas & Sampling
# -----------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = "\nROMEO:\nWhat light "
    max_new_tokens: int = 60
    temperature: float = 0.7
    top_k: int = 5


@torch.no_grad()
def sample_tokens(model: NanoGPT, prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    tokens = encode(prompt)
    if not tokens:
        tokens = [0]

    idx = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return decode(idx[0].tolist())


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.post("/generate")
def generate(req: GenerateRequest):
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    output = sample_tokens(model_instance, req.prompt, req.max_new_tokens, req.temperature, req.top_k)
    return {"generated_text": output}


# -----------------------------------------------------------------------------
# Webpage directly built into app.py (No static folder needed)
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>NanoGPT Interactive Console</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col items-center p-6">
      <div class="w-full max-w-2xl bg-slate-800 rounded-xl shadow-lg border border-slate-700 p-6 space-y-5">
        <h1 class="text-2xl font-bold text-white">NanoGPT DPO Console</h1>
        <p class="text-sm text-slate-400">Direct Preference Optimization Inference</p>

        <div>
          <label class="block text-sm font-medium mb-1">Prompt</label>
          <textarea id="prompt" rows="3" class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-slate-100 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500">\nROMEO:\nWhat light </textarea>
        </div>

        <div class="grid grid-cols-3 gap-3">
          <div>
            <label class="text-xs text-slate-400">Max Tokens</label>
            <input id="maxTokens" type="number" value="60" class="w-full bg-slate-950 border border-slate-700 rounded p-2 text-sm">
          </div>
          <div>
            <label class="text-xs text-slate-400">Temperature</label>
            <input id="temp" type="number" step="0.1" value="0.7" class="w-full bg-slate-950 border border-slate-700 rounded p-2 text-sm">
          </div>
          <div>
            <label class="text-xs text-slate-400">Top-K</label>
            <input id="topK" type="number" value="5" class="w-full bg-slate-950 border border-slate-700 rounded p-2 text-sm">
          </div>
        </div>

        <button onclick="generateText()" id="btn" class="w-full bg-blue-600 hover:bg-blue-500 py-2.5 rounded-lg font-semibold transition">
          Generate Continuation
        </button>

        <div>
          <label class="block text-sm font-medium mb-1">Output</label>
          <pre id="output" class="bg-slate-950 border border-slate-700 rounded-lg p-4 font-mono text-sm text-emerald-400 whitespace-pre-wrap min-h-[100px]">Waiting for generation...</pre>
        </div>
      </div>

      <script>
        async function generateText() {
          const btn = document.getElementById('btn');
          const output = document.getElementById('output');
          btn.disabled = true;
          btn.innerText = "Generating...";
          output.innerText = "Processing tokens...";

          try {
            const res = await fetch('/generate', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                prompt: document.getElementById('prompt').value,
                max_new_tokens: parseInt(document.getElementById('maxTokens').value),
                temperature: parseFloat(document.getElementById('temp').value),
                top_k: parseInt(document.getElementById('topK').value)
              })
            });
            const data = await res.json();
            output.innerText = data.generated_text;
          } catch (e) {
            output.innerText = "Error: " + e.message;
          } finally {
            btn.disabled = false;
            btn.innerText = "Generate Continuation";
          }
        }
      </script>
    </body>
    </html>
    """