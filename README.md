# Question Answering App

The FastAPI app sends questions to an OpenAI-compatible hosted chat-completions
API. The local NanoGPT checkpoints are not used by this app; they were trained
on Shakespeare and are not suitable for general question answering.

## Run locally on Windows

In PowerShell, set your API key for the current terminal session and start the
server:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

The default provider URL is `https://api.openai.com/v1` and the default model is
`gpt-4o-mini`. To use another OpenAI-compatible provider or model, set
`OPENAI_BASE_URL` and `OPENAI_MODEL` before starting the server. Keep the API
key in the server environment; do not put it in browser code or commit it.
