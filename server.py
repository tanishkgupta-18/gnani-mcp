"""Gnani MCP Server for AgenticOrg.

Exposes production-ready Gnani Speech-to-Text (STT) and Text-to-Speech (TTS)
tools via the Model Context Protocol (MCP) using HTTP/SSE and Streamable HTTP transports.
"""

import base64
from datetime import datetime, timezone
import logging
import os
import sys
from typing import Any
import uuid

from dotenv import load_dotenv
import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
import uvicorn

# Load environment variables from .env if present
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gnani-mcp")

# Import MCP Server (supports mcp 2.x and backwards-compatible imports)
try:
    from mcp.server.mcpserver import MCPServer
except (ImportError, ModuleNotFoundError):
    from mcp.server.fastmcp import FastMCP as MCPServer

from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import (
    StreamableHTTPASGIApp,
    StreamableHTTPSessionManager,
)
from mcp.server.transport_security import TransportSecuritySettings

# Gnani API Endpoints
GNANI_STT_URL = "https://api.vachana.ai/stt/v3"
GNANI_TTS_URL = "https://api.gnani.ai/v1/tts/inference"

# Initialize FastMCP / MCPServer instance
mcp_server = MCPServer("Gnani MCP")


def _get_stt_api_key() -> str | None:
    """Retrieve Gnani STT API Key from environment."""
    return os.environ.get("GNANI_STT_API_KEY") or os.environ.get("GNANI_API_KEY")


def _get_tts_api_key() -> str | None:
    """Retrieve Gnani TTS API Key from environment."""
    return os.environ.get("GNANI_TTS_API_KEY") or os.environ.get("GNANI_API_KEY")


def _mask_key(key: str | None) -> str:
    """Safely mask key for display without exposing credentials."""
    if not key:
        return "[NOT CONFIGURED]"
    if len(key) <= 6:
        return "******"
    return f"{key[:3]}...{key[-3:]}"


# ==============================================================================
# Tool 1: gnani_speech_to_text
# ==============================================================================
@mcp_server.tool(
    name="gnani_speech_to_text",
    description=(
        "Transcribe audio using Gnani's real Speech-to-Text (STT) API (https://api.vachana.ai/stt/v3). "
        "Decodes base64-encoded audio and returns a structured transcription response."
    ),
)
async def gnani_speech_to_text(
    audio_base64: str,
    filename: str = "audio.wav",
    language_code: str = "en-IN",
    preferred_language: str = "en-IN",
    format: str = "transcribe",
    itn_native_numerals: bool = True,
) -> dict[str, Any]:
    """Transcribe audio bytes using Gnani Vachana STT v3 API.

    Args:
        audio_base64: Base64-encoded audio data string.
        filename: Name of the audio file (default: 'audio.wav').
        language_code: Target language code (default: 'en-IN').
        preferred_language: Preferred spoken language code (default: 'en-IN').
        format: Transcription format mode (default: 'transcribe').
        itn_native_numerals: Inverse text normalization for native numerals (default: True).

    Returns:
        Structured dictionary with success status, request_id, timestamp, transcript,
        and raw response or error details.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Validate credentials
    api_key = _get_stt_api_key()
    if not api_key:
        logger.warning("gnani_speech_to_text invoked without GNANI_STT_API_KEY configured")
        return {
            "success": False,
            "status_code": None,
            "error_type": "missing_credentials",
            "error": "GNANI_STT_API_KEY environment variable is not configured. Please set GNANI_STT_API_KEY.",
            "request_id": None,
            "timestamp": now_iso,
            "transcript": None,
            "raw_response": {"detail": "Missing GNANI_STT_API_KEY environment variable"},
        }

    # 2. Decode base64 audio
    clean_base64 = audio_base64.strip()
    if "," in clean_base64 and clean_base64.startswith("data:"):
        clean_base64 = clean_base64.split(",", 1)[1]

    try:
        audio_bytes = base64.b64decode(clean_base64)
        if len(audio_bytes) == 0:
            raise ValueError("Decoded audio payload is empty")
    except Exception as e:
        logger.error(f"Failed to decode base64 audio: {e}")
        return {
            "success": False,
            "status_code": 400,
            "error_type": "invalid_audio_input",
            "error": f"Invalid base64 audio data: {e}",
            "request_id": None,
            "timestamp": now_iso,
            "transcript": None,
            "raw_response": {"detail": str(e)},
        }

    # 3. Prepare multipart/form-data payload
    mime_type = "audio/wav"
    lower_fn = (filename or "").lower()
    if lower_fn.endswith(".mp3"):
        mime_type = "audio/mpeg"
    elif lower_fn.endswith(".ogg"):
        mime_type = "audio/ogg"
    elif lower_fn.endswith(".flac"):
        mime_type = "audio/flac"

    files = {
        "audio_file": (filename or "audio.wav", audio_bytes, mime_type),
    }
    data = {
        "language_code": language_code,
        "preferred_language": preferred_language,
        "format": format,
        "itn_native_numerals": "true" if itn_native_numerals else "false",
    }
    headers = {
        "X-API-Key-ID": api_key,
    }

    # 4. Invoke Gnani STT API
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                GNANI_STT_URL,
                files=files,
                data=data,
                headers=headers,
            )
    except httpx.TimeoutException as e:
        logger.error(f"Gnani STT API timeout: {e}")
        return {
            "success": False,
            "status_code": 408,
            "error_type": "timeout_error",
            "error": f"Gnani STT API request timed out: {e}",
            "request_id": None,
            "timestamp": now_iso,
            "transcript": None,
            "raw_response": {"detail": "Request timed out", "error": str(e)},
        }
    except httpx.RequestError as e:
        logger.error(f"Gnani STT API connection failure: {e}")
        return {
            "success": False,
            "status_code": None,
            "error_type": "network_error",
            "error": f"Failed to connect to Gnani STT API: {e}",
            "request_id": None,
            "timestamp": now_iso,
            "transcript": None,
            "raw_response": {"detail": "Network connection error", "error": str(e)},
        }

    # 5. Parse response
    status_code = response.status_code
    try:
        raw_response = response.json()
    except Exception:
        raw_response = {"raw_text": response.text[:2000]}

    request_id = (
        (raw_response.get("request_id") if isinstance(raw_response, dict) else None)
        or (raw_response.get("requestId") if isinstance(raw_response, dict) else None)
        or response.headers.get("x-request-id")
        or str(uuid.uuid4())
    )
    resp_timestamp = (
        (raw_response.get("timestamp") if isinstance(raw_response, dict) else None)
        or (raw_response.get("time") if isinstance(raw_response, dict) else None)
        or now_iso
    )

    if 200 <= status_code < 300:
        # Check if Gnani returned an application-level error inside 2xx
        if isinstance(raw_response, dict) and raw_response.get("success") is False:
            err_msg = raw_response.get("error") or raw_response.get("message") or "STT API reported error"
            return {
                "success": False,
                "status_code": status_code,
                "error_type": "api_error",
                "error": str(err_msg),
                "request_id": request_id,
                "timestamp": resp_timestamp,
                "transcript": None,
                "raw_response": raw_response,
            }

        # Extract transcript without inventing data
        transcript = ""
        if isinstance(raw_response, dict):
            if "transcript" in raw_response and isinstance(raw_response["transcript"], str):
                transcript = raw_response["transcript"]
            elif "transcription" in raw_response and isinstance(raw_response["transcription"], str):
                transcript = raw_response["transcription"]
            elif "text" in raw_response and isinstance(raw_response["text"], str):
                transcript = raw_response["text"]
            elif "data" in raw_response and isinstance(raw_response["data"], dict):
                transcript = (
                    raw_response["data"].get("transcript")
                    or raw_response["data"].get("transcription")
                    or raw_response["data"].get("text")
                    or ""
                )
            elif "results" in raw_response:
                transcript = str(raw_response["results"])

        return {
            "success": True,
            "request_id": str(request_id),
            "timestamp": str(resp_timestamp),
            "transcript": transcript,
            "raw_response": raw_response,
        }

    # Error handling for >= 400
    if status_code in (401, 403):
        error_type = "authentication_error"
    elif status_code in (400, 422):
        error_type = "invalid_request"
    elif status_code == 429:
        error_type = "rate_limited"
    elif 500 <= status_code < 600:
        error_type = "upstream_gnani_failure"
    else:
        error_type = f"http_error_{status_code}"

    error_msg = f"Gnani STT API failed with HTTP {status_code}: {error_type}"
    if isinstance(raw_response, dict):
        detail = raw_response.get("message") or raw_response.get("error") or raw_response.get("detail")
        if detail:
            error_msg = f"{error_msg} - {detail}"

    logger.error(f"Gnani STT error: {error_msg}")
    return {
        "success": False,
        "status_code": status_code,
        "error_type": error_type,
        "error": error_msg,
        "request_id": str(request_id),
        "timestamp": str(resp_timestamp),
        "transcript": None,
        "raw_response": raw_response,
    }


# ==============================================================================
# Tool 2: gnani_text_to_speech
# ==============================================================================
@mcp_server.tool(
    name="gnani_text_to_speech",
    description=(
        "Synthesize speech from text using Gnani's real Text-to-Speech (TTS) API (https://api.gnani.ai/v1/tts/inference). "
        "Returns synthesized audio as base64 without writing permanently to disk."
    ),
)
async def gnani_text_to_speech(
    text: str,
    language: str = "en-IN",
    voice: str = "Yashvi",
    sample_rate: int = 48000,
) -> dict[str, Any]:
    """Convert text into speech using Gnani TTS API.

    Args:
        text: Text to synthesize.
        language: Language code (default: 'en-IN').
        voice: Voice name (default: 'Yashvi').
        sample_rate: Audio sample rate in Hz (default: 48000).

    Returns:
        Structured dictionary with success status, base64-encoded audio,
        content_type, sample_rate, and raw metadata or error details.
    """
    # 1. Validate credentials
    api_key = _get_tts_api_key()
    if not api_key:
        logger.warning("gnani_text_to_speech invoked without GNANI_TTS_API_KEY configured")
        return {
            "success": False,
            "status_code": None,
            "error_type": "missing_credentials",
            "error": "GNANI_TTS_API_KEY environment variable is not configured. Please set GNANI_TTS_API_KEY.",
            "raw_metadata": {"detail": "Missing GNANI_TTS_API_KEY environment variable"},
        }

    # 2. Prepare payload
    payload = {
        "model": "timbre-2.5",
        "language": language,
        "voice": voice,
        "sample_rate": sample_rate,
        "text": text,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # 3. Invoke Gnani TTS API
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                GNANI_TTS_URL,
                json=payload,
                headers=headers,
            )
    except httpx.TimeoutException as e:
        logger.error(f"Gnani TTS API timeout: {e}")
        return {
            "success": False,
            "status_code": 408,
            "error_type": "timeout_error",
            "error": f"Gnani TTS API request timed out: {e}",
            "raw_metadata": {"detail": "Request timed out", "error": str(e)},
        }
    except httpx.RequestError as e:
        logger.error(f"Gnani TTS API connection failure: {e}")
        return {
            "success": False,
            "status_code": None,
            "error_type": "network_error",
            "error": f"Failed to connect to Gnani TTS API: {e}",
            "raw_metadata": {"detail": "Network connection error", "error": str(e)},
        }

    status_code = response.status_code
    content_type_header = response.headers.get("content-type", "").lower()

    if 200 <= status_code < 300:
        # Check if response returned binary audio directly
        if "audio" in content_type_header or "octet-stream" in content_type_header:
            audio_bytes = response.content
            if len(audio_bytes) == 0:
                return {
                    "success": False,
                    "status_code": status_code,
                    "error_type": "empty_audio",
                    "error": "Gnani TTS API returned empty audio stream",
                    "raw_metadata": {"headers": dict(response.headers)},
                }
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            return {
                "success": True,
                "audio_base64": audio_base64,
                "content_type": response.headers.get("content-type", "audio/wav"),
                "sample_rate": sample_rate,
                "raw_metadata": {
                    "size_bytes": len(audio_bytes),
                    "headers": {k: v for k, v in response.headers.items() if k.lower() != "authorization"},
                },
            }

        # Otherwise parse JSON response
        try:
            raw_json = response.json()
        except Exception:
            raw_json = {"raw_text": response.text[:2000]}

        audio_base64 = None
        if isinstance(raw_json, dict):
            if "audio" in raw_json and isinstance(raw_json["audio"], str):
                audio_base64 = raw_json["audio"]
            elif "audio_base64" in raw_json and isinstance(raw_json["audio_base64"], str):
                audio_base64 = raw_json["audio_base64"]
            elif "audioContent" in raw_json and isinstance(raw_json["audioContent"], str):
                audio_base64 = raw_json["audioContent"]
            elif "data" in raw_json and isinstance(raw_json["data"], dict):
                audio_base64 = (
                    raw_json["data"].get("audio")
                    or raw_json["data"].get("audio_base64")
                    or raw_json["data"].get("audioContent")
                )

        if audio_base64:
            clean_b64 = audio_base64.strip()
            if "," in clean_b64 and clean_b64.startswith("data:"):
                clean_b64 = clean_b64.split(",", 1)[1]
            return {
                "success": True,
                "audio_base64": clean_b64,
                "content_type": (
                    raw_json.get("content_type")
                    or (raw_json.get("data", {}).get("content_type") if isinstance(raw_json.get("data"), dict) else None)
                    or "audio/wav"
                ),
                "sample_rate": sample_rate,
                "raw_metadata": raw_json,
            }

        # Status 200 but no audio provided (or error payload inside JSON)
        return {
            "success": False,
            "status_code": status_code,
            "error_type": "missing_audio_payload",
            "error": "Gnani TTS API returned 200 OK but audio data was not found in response",
            "raw_metadata": raw_json,
        }

    # Error handling for >= 400
    if status_code in (401, 403):
        error_type = "authentication_error"
    elif status_code in (400, 422):
        error_type = "invalid_request"
    elif status_code == 429:
        error_type = "rate_limited"
    elif 500 <= status_code < 600:
        error_type = "upstream_gnani_failure"
    else:
        error_type = f"http_error_{status_code}"

    try:
        raw_metadata = response.json()
    except Exception:
        raw_metadata = {"raw_text": response.text[:2000]}

    error_msg = f"Gnani TTS API failed with HTTP {status_code}: {error_type}"
    if isinstance(raw_metadata, dict):
        detail = raw_metadata.get("message") or raw_metadata.get("error") or raw_metadata.get("detail")
        if detail:
            error_msg = f"{error_msg} - {detail}"

    logger.error(f"Gnani TTS error: {error_msg}")
    return {
        "success": False,
        "status_code": status_code,
        "error_type": error_type,
        "error": error_msg,
        "raw_metadata": raw_metadata,
    }


# ==============================================================================
# Tool 3: health_check
# ==============================================================================
@mcp_server.tool(
    name="health_check",
    description="Check the health status of the Gnani MCP service.",
)
async def health_check() -> dict[str, Any]:
    """MCP tool to verify service health."""
    return {
        "status": "ok",
        "service": "Gnani MCP",
    }


# ==============================================================================
# Application Factory: Starlette App with SSE, Streamable HTTP & Health Check
# ==============================================================================
def create_app() -> Starlette:
    """Build the production Starlette app with HTTP endpoints and MCP transports."""
    raw_allowed_hosts = os.environ.get("ALLOWED_HOSTS", "*").strip()

    if raw_allowed_hosts == "*" or not raw_allowed_hosts:
        # Allow all hosts (e.g. Render dynamic domains, local development)
        sec_settings = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        logger.info("Host security: DNS rebinding protection disabled (ALLOWED_HOSTS='*')")
    else:
        hosts_list = [h.strip() for h in raw_allowed_hosts.split(",") if h.strip()]
        allowed_hosts: list[str] = []
        allowed_origins: list[str] = []
        for h in hosts_list:
            allowed_hosts.append(h)
            if not h.endswith(":*"):
                allowed_hosts.append(f"{h}:*")
            allowed_origins.extend([
                f"http://{h}",
                f"https://{h}",
                f"http://{h}:*",
                f"https://{h}:*",
            ])

        # Always include localhost and loopback for local health checks and testing
        for local in ("localhost", "127.0.0.1", "[::1]"):
            if local not in allowed_hosts:
                allowed_hosts.extend([local, f"{local}:*"])
                allowed_origins.extend([f"http://{local}", f"http://{local}:*", f"https://{local}:*"])

        sec_settings = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
        )
        logger.info(f"Host security: DNS rebinding enabled with allowed hosts: {allowed_hosts}")

    # Set up SSE Transport
    sse_transport = SseServerTransport("/messages/", security_settings=sec_settings)

    async def handle_sse(request: Request) -> Response:
        async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
            await mcp_server._lowlevel_server.run(
                streams[0],
                streams[1],
                mcp_server._lowlevel_server.create_initialization_options(),
            )
        return Response()

    # Set up Streamable HTTP Transport
    session_manager = StreamableHTTPSessionManager(
        app=mcp_server._lowlevel_server,
        security_settings=sec_settings,
    )
    streamable_app = StreamableHTTPASGIApp(session_manager)

    # Health check HTTP endpoint for Render and uptime monitoring
    async def render_health_endpoint(request: Request) -> JSONResponse:
        return JSONResponse({
            "status": "ok",
            "service": "Gnani MCP",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    routes = [
        # HTTP health check endpoint
        Route("/health", endpoint=render_health_endpoint, methods=["GET"]),
        # SSE transport endpoints
        Route("/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/messages", app=sse_transport.handle_post_message),
        # Streamable HTTP MCP transport endpoint
        Route("/mcp", endpoint=streamable_app, methods=["GET", "POST", "DELETE"]),
    ]

    app = Starlette(
        debug=os.environ.get("DEBUG", "false").lower() == "true",
        routes=routes,
        lifespan=lambda _app: session_manager.run(),
    )
    return app


app = create_app()


def main():
    port = int(os.environ.get("PORT", "8000"))
    host = "0.0.0.0"

    print("==================================================")
    print("           Gnani MCP Server for AgenticOrg        ")
    print("==================================================")
    print(f" Binding:              http://{host}:{port}")
    print(f" Render Health URL:    http://{host}:{port}/health")
    print(f" MCP SSE Endpoint:     http://{host}:{port}/sse")
    print(f" MCP Streamable HTTP:  http://{host}:{port}/mcp")
    print(f" STT API Key:          {_mask_key(_get_stt_api_key())}")
    print(f" TTS API Key:          {_mask_key(_get_tts_api_key())}")
    print(f" Allowed Hosts:        {os.environ.get('ALLOWED_HOSTS', '*')}")
    print("==================================================")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
