
import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Gemini Question Answering App")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=12000)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.2, ge=0, le=2)


def get_answer(req: GenerateRequest) -> str:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key missing. Set GEMINI_API_KEY in your .env file.",
        )

    payload = {
        "systemInstruction": {
            "parts": [
                {
                    "text": (
                        "Answer the user's question directly and clearly. "
                        "Be accurate and concise. If unsure, say so instead "
                        "of inventing an answer."
                    )
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": req.prompt}],
            }
        ],
        "generationConfig": {
            "maxOutputTokens": req.max_new_tokens,
            "temperature": req.temperature,
        },
    }

    request = Request(
        GEMINI_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))

    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logger.error("Gemini API HTTP %s: %s", exc.code, error_body[:1000])

        if exc.code == 400:
            detail = "Gemini rejected the request. Check the model name and request settings."
        elif exc.code == 401 or exc.code == 403:
            detail = (
                "Gemini authentication failed. Check your API key, "
                "key restrictions, and Gemini API access."
            )
        elif exc.code == 404:
            detail = "Gemini model not found. Check the GEMINI_MODEL setting."
        elif exc.code == 429:
            detail = "Gemini quota or rate limit reached. Check your API usage and billing limits."
        else:
            detail = f"Gemini API request failed with HTTP {exc.code}."

        raise HTTPException(status_code=502, detail=detail) from exc

    except (URLError, TimeoutError) as exc:
        logger.exception("Could not connect to Gemini")
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Google Gemini. Check your internet connection.",
        ) from exc

    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.exception("Gemini returned invalid JSON")
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an invalid response.",
        ) from exc

    try:
        candidate = result["candidates"][0]
        parts = candidate["content"]["parts"]
        answer = "\n".join(
            part["text"] for part in parts if isinstance(part.get("text"), str)
        )
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        logger.error("Unexpected Gemini response: %s", result)
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini returned no usable answer. The response may have "
                "been blocked or stopped without text."
            ),
        ) from exc

    if not answer.strip():
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty answer.",
        )

    return answer.strip()


@app.post("/generate")
def generate(req: GenerateRequest):
    return {"generated_text": get_answer(req)}


@app.get("/", response_class=HTMLResponse)
def serve_home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Gemini Question Answering</title>
      <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-slate-900 text-slate-100 min-h-screen flex flex-col items-center p-6">
      <div class="w-full max-w-2xl bg-slate-800 rounded-xl shadow-lg border border-slate-700 p-6 space-y-5">
        <h1 class="text-2xl font-bold text-white">Ask a Question</h1>
        <p class="text-sm text-slate-400">Get a clear answer powered by Google Gemini.</p>

        <div>
          <label class="block text-sm font-medium mb-1" for="prompt">Your question</label>
          <textarea id="prompt" rows="3" maxlength="12000"
            class="w-full bg-slate-950 border border-slate-700 rounded-lg p-3 text-slate-100 text-sm"
            placeholder="For example: What is the capital of Andhra Pradesh?"></textarea>
        </div>

        <div class="grid grid-cols-2 gap-3">
          <div>
            <label class="text-xs text-slate-400" for="maxTokens">Maximum answer length</label>
            <input id="maxTokens" type="number" min="1" max="2048" value="256"
              class="w-full bg-slate-950 border border-slate-700 rounded p-2 text-sm">
          </div>
          <div>
            <label class="text-xs text-slate-400" for="temp">Temperature</label>
            <input id="temp" type="number" min="0" max="2" step="0.1" value="0.2"
              class="w-full bg-slate-950 border border-slate-700 rounded p-2 text-sm">
          </div>
        </div>

        <button onclick="generateText()" id="btn"
          class="w-full bg-blue-600 hover:bg-blue-500 py-2.5 rounded-lg font-semibold">
          Get Answer
        </button>

        <div>
          <label class="block text-sm font-medium mb-1" for="output">Answer</label>
          <pre id="output"
            class="bg-slate-950 border border-slate-700 rounded-lg p-4 text-sm text-emerald-400 whitespace-pre-wrap min-h-[100px]">Your answer will appear here.</pre>
        </div>
      </div>

      <script>
        async function generateText() {
          const btn = document.getElementById("btn");
          const output = document.getElementById("output");
          btn.disabled = true;
          btn.innerText = "Thinking...";
          output.innerText = "Getting an answer...";

          try {
            const res = await fetch("/generate", {
              method: "POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({
                prompt: document.getElementById("prompt").value,
                max_new_tokens: Number(document.getElementById("maxTokens").value),
                temperature: Number(document.getElementById("temp").value)
              })
            });

            const data = await res.json();
            if (!res.ok) {
              throw new Error(data.detail || `Request failed (${res.status})`);
            }
            output.innerText = data.generated_text;
          } catch (e) {
            output.innerText = "Error: " + e.message;
          } finally {
            btn.disabled = false;
            btn.innerText = "Get Answer";
          }
        }
      </script>
    </body>
    </html>
    """
