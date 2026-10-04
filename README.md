# Gnani MCP Server

A minimal, production-ready Python **Model Context Protocol (MCP)** server for **AgenticOrg**.

> **Important**: This MCP server is **NOT** an AI agent. AgenticOrg is the autonomous agent. This server acts strictly as an MCP tool provider that exposes Gnani's speech tools and calls Gnani's real APIs over HTTP.

---

## Architecture & Transports

- **SSE Transport**: Mounted at `/sse` (with `/messages/` endpoint for incoming messages). Fully compatible with AgenticOrg and MCP SSE clients.
- **Streamable HTTP Transport**: Mounted at `/mcp` for modern streamable HTTP MCP clients.
- **Render Health Check Endpoint**: `GET /health` returns JSON `{"status": "ok", "service": "Gnani MCP"}` for Render uptime and deployment health monitoring.
- **Security & DNS Rebinding**: Protects against DNS rebinding attacks while allowing both local development and Render hostnames via `ALLOWED_HOSTS`.

---

## MCP Tools

### 1. `gnani_speech_to_text`
Transcribes audio using Gnani's real Speech-to-Text API (`POST https://api.vachana.ai/stt/v3`).

- **Inputs**:
  - `audio_base64` (*string*, required): Base64-encoded audio bytes.
  - `filename` (*string*, optional, default: `"audio.wav"`): Audio file name with extension (`.wav`, `.mp3`, `.ogg`, `.flac`).
  - `language_code` (*string*, optional, default: `"en-IN"`): Language code.
  - `preferred_language` (*string*, optional, default: `"en-IN"`): Preferred spoken language code.
  - `format` (*string*, optional, default: `"transcribe"`): Transcription mode.
  - `itn_native_numerals` (*boolean*, optional, default: `true`): Inverse text normalization for native numerals.
- **Headers Sent**: `X-API-Key-ID: <GNANI_STT_API_KEY>`
- **Output**:
  ```json
  {
    "success": true,
    "request_id": "...",
    "timestamp": "2026-10-04T12:00:00+00:00",
    "transcript": "Transcribed text...",
    "raw_response": { ... }
  }
  ```
- **Error Behavior**: If the API fails or credentials are missing, returns `success: false` with HTTP status code and error details. Transcripts are **never** invented.

---

### 2. `gnani_text_to_speech`
Synthesizes speech from text using Gnani's real Text-to-Speech API (`POST https://api.gnani.ai/v1/tts/inference`).

- **Inputs**:
  - `text` (*string*, required): Text to convert into speech.
  - `language` (*string*, optional, default: `"en-IN"`): Target language code.
  - `voice` (*string*, optional, default: `"Yashvi"`): Voice model name.
  - `sample_rate` (*integer*, optional, default: `48000`): Audio sample rate in Hz.
- **Headers Sent**:
  - `Authorization: Bearer <GNANI_TTS_API_KEY>`
  - `Content-Type: application/json`
- **Output**:
  ```json
  {
    "success": true,
    "audio_base64": "UklGRi...",
    "content_type": "audio/wav",
    "sample_rate": 48000,
    "raw_metadata": { ... }
  }
  ```
- **Memory & Storage**: Audio is handled entirely in-memory and returned as base64 without writing permanently to disk.
- **Error Behavior**: If the API fails, returns `success: false` with status code and error details. Audio is **never** fabricated.

---

### 3. `health_check`
No inputs required. Verifies service availability.

- **Output**:
  ```json
  {
    "status": "ok",
    "service": "Gnani MCP"
  }
  ```

---

## Local Development & Setup

### Prerequisites
- Python 3.10+ installed

### 1. Create Virtual Environment & Install Dependencies

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Configure your credentials:
```env
GNANI_STT_API_KEY=your_gnani_stt_api_key_here
GNANI_TTS_API_KEY=your_gnani_tts_api_key_here

PORT=8000
ALLOWED_HOSTS=*
```

> **Note**: Real credentials are **not** required to start the server. If credentials are unset, the server starts successfully and tools return a clear "credential not configured" error only when invoked. Full credentials are never logged.

### 3. Run the Server Locally

```bash
python server.py
```

The server binds to `0.0.0.0:8000`:
- **Local Health URL**: `http://127.0.0.1:8000/health`
- **Local MCP SSE URL**: `http://127.0.0.1:8000/sse`
- **Local MCP Streamable HTTP**: `http://127.0.0.1:8000/mcp`

---

## Verification & Testing

Run the included automated test suite against the running server:

```bash
python test_server.py
```

The test script validates:
1. `GET /health` endpoint (HTTP 200 OK)
2. MCP Session Initialization & Tool Discovery over SSE transport
3. `health_check` tool call
4. `gnani_speech_to_text` tool call with generated sample WAV audio
5. `gnani_text_to_speech` tool call with `"I found a difference in the waiting period between the two policies."`
6. Streamable HTTP transport at `/mcp`

---

## Deployment to Render

1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your Git repository (or push this package).
3. Set the service settings:
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     python server.py
     ```
   - **Health Check Path**: `/health`
4. Set **Environment Variables** in Render Dashboard:
   - `GNANI_STT_API_KEY`: Your Gnani STT API Key
   - `GNANI_TTS_API_KEY`: Your Gnani TTS API Key
   - `ALLOWED_HOSTS`: `*` (or your Render service domain, e.g. `your-service.onrender.com,localhost`)

Render will automatically assign `PORT` (e.g., 10000), which `server.py` reads automatically from the environment.

---

## Registering in AgenticOrg

Once deployed, register the MCP server in AgenticOrg using the SSE transport URL:

- **SSE URL**: `https://<your-render-service>.onrender.com/sse`
- **Transport**: `sse`
- **Exposed Tools**:
  - `gnani_speech_to_text`
  - `gnani_text_to_speech`
  - `health_check`

---

## Project Structure

```
gnani-mcp/
├── .env.example       # Example environment variables template
├── .gitignore          # Git ignore for venv, .env, __pycache__, and archives
├── requirements.txt    # Project dependencies
├── server.py           # Production MCP server implementation
├── test_server.py      # Automated test and verification suite
└── README.md           # Documentation and deployment guide
```
