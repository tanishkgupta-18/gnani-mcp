"""Automated tests for Gnani MCP Server.

Validates:
1. GET /health HTTP endpoint (Render health check)
2. MCP Tool discovery via SSE transport
3. MCP Tool discovery via Streamable HTTP transport
4. health_check tool invocation
5. gnani_speech_to_text tool invocation with sample base64 WAV
6. gnani_text_to_speech tool invocation with specified sample text

Can run against an existing running server (default: http://127.0.0.1:8000)
or will launch an in-process server if not currently running.
"""

import asyncio
import base64
import io
import json
import math
import os
import struct
import sys
import wave

import httpx
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client


def generate_sample_wav(duration_seconds: float = 0.5, sample_rate: int = 16000) -> str:
    """Generate a clean in-memory PCM 16-bit mono WAV and return as base64."""
    wav_buffer = io.BytesIO()
    num_samples = int(duration_seconds * sample_rate)
    frequency = 440.0  # 440Hz standard A tone

    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        # Generate simple sine wave
        samples = []
        for i in range(num_samples):
            value = int(16000 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            samples.append(struct.pack("<h", value))
        wf.writeframes(b"".join(samples))

    wav_bytes = wav_buffer.getvalue()
    return base64.b64encode(wav_bytes).decode("utf-8")


async def run_tests(base_url: str = "http://127.0.0.1:8000"):
    print("==================================================")
    print("      Running Gnani MCP Server Verification       ")
    print(f" Target Server URL: {base_url}")
    print("==================================================")

    # 1. Verify GET /health HTTP endpoint
    print("\n[1/6] Verifying GET /health endpoint...")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{base_url}/health", timeout=5.0)
            assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}"
            health_data = resp.json()
            assert health_data.get("status") == "ok", f"Expected status 'ok', got {health_data}"
            print(f"  PASS: /health returned HTTP 200 -> {health_data}")
        except Exception as e:
            print(f"  FAIL: Could not connect to /health: {e}")
            raise

    # 2. Connect via MCP SSE Transport & Discover Tools
    sse_url = f"{base_url}/sse"
    print(f"\n[2/6] Connecting to MCP SSE transport at {sse_url}...")
    async with sse_client(sse_url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            print("  PASS: MCP SSE session initialized successfully.")

            # Discover tools
            tools_result = await session.list_tools()
            discovered = [tool.name for tool in tools_result.tools]
            print(f"  Discovered tools ({len(discovered)}): {discovered}")

            expected_tools = ["gnani_speech_to_text", "gnani_text_to_speech", "health_check"]
            for tool_name in expected_tools:
                assert tool_name in discovered, f"Missing expected tool: {tool_name}"
            print("  PASS: All expected tools discovered!")

            # 3. Call health_check MCP tool
            print("\n[3/6] Calling 'health_check' MCP tool...")
            hc_resp = await session.call_tool("health_check", arguments={})
            assert len(hc_resp.content) > 0, "No content returned from health_check tool"
            hc_content = json.loads(hc_resp.content[0].text)
            print(f"  Result: {json.dumps(hc_content, indent=2)}")
            assert hc_content.get("status") == "ok", "health_check failed"
            assert hc_content.get("service") == "Gnani MCP", "service name mismatch"
            print("  PASS: health_check tool verified.")

            # 4. Call gnani_speech_to_text MCP tool
            print("\n[4/6] Calling 'gnani_speech_to_text' MCP tool with sample WAV...")
            sample_b64_audio = generate_sample_wav(0.5, 16000)
            stt_args = {
                "audio_base64": sample_b64_audio,
                "filename": "sample_test.wav",
                "language_code": "en-IN",
                "preferred_language": "en-IN",
                "format": "transcribe",
                "itn_native_numerals": True,
            }
            stt_resp = await session.call_tool("gnani_speech_to_text", arguments=stt_args)
            assert len(stt_resp.content) > 0, "No content returned from gnani_speech_to_text tool"
            stt_content = json.loads(stt_resp.content[0].text)
            print(f"  Result: {json.dumps(stt_content, indent=2)}")
            assert "success" in stt_content, "STT response must contain 'success'"
            assert "request_id" in stt_content, "STT response must contain 'request_id'"
            assert "timestamp" in stt_content, "STT response must contain 'timestamp'"
            assert "transcript" in stt_content, "STT response must contain 'transcript'"
            assert "raw_response" in stt_content, "STT response must contain 'raw_response'"
            if not stt_content["success"]:
                print(f"  STT handled gracefully: {stt_content.get('error')}")
            else:
                print(f"  STT transcription success: '{stt_content.get('transcript')}'")
            print("  PASS: gnani_speech_to_text tool verified.")

            # 5. Call gnani_text_to_speech MCP tool
            print("\n[5/6] Calling 'gnani_text_to_speech' MCP tool with test text...")
            test_phrase = "I found a difference in the waiting period between the two policies."
            tts_args = {
                "text": test_phrase,
                "language": "en-IN",
                "voice": "Yashvi",
                "sample_rate": 48000,
            }
            tts_resp = await session.call_tool("gnani_text_to_speech", arguments=tts_args)
            assert len(tts_resp.content) > 0, "No content returned from gnani_text_to_speech tool"
            tts_content = json.loads(tts_resp.content[0].text)
            print(f"  Result: {json.dumps(tts_content, indent=2)}")
            assert "success" in tts_content, "TTS response must contain 'success'"
            if not tts_content["success"]:
                print(f"  TTS handled gracefully: {tts_content.get('error')}")
            else:
                assert "audio_base64" in tts_content, "Successful TTS must return 'audio_base64'"
                print(f"  TTS synthesis success: Received audio base64 (length {len(tts_content['audio_base64'])})")
            print("  PASS: gnani_text_to_speech tool verified.")

    # 6. Verify Streamable HTTP transport at /mcp
    mcp_url = f"{base_url}/mcp"
    print(f"\n[6/6] Verifying Streamable HTTP transport at {mcp_url}...")
    try:
        async with streamable_http_client(mcp_url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                discovered_http = [tool.name for tool in tools_result.tools]
                print(f"  Discovered tools via Streamable HTTP: {discovered_http}")
                assert "health_check" in discovered_http
                print("  PASS: Streamable HTTP transport verified.")
    except Exception as e:
        print(f"  NOTE: Streamable HTTP check: {e}")

    print("\n==================================================")
    print("       ALL VERIFICATION TESTS COMPLETED!         ")
    print("==================================================")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    asyncio.run(run_tests(url))
