"""
AITorrent Quickstart — 2-peer distributed inference demo.

Usage:
  Terminal 1 (Peer A — serves layers 0 to N/2):
    aitorrent serve meta-llama/Llama-3.1-8B --layers 0:16 --port 9877

  Terminal 2 (Peer B — serves layers N/2 to N, connects to Peer A):
    aitorrent serve meta-llama/Llama-3.1-8B --layers 16:32 --port 9878 --peer localhost:9877

  Terminal 3 (Client — sends inference request):
    aitorrent infer "Explain quantum computing in simple terms" --model meta-llama/Llama-3.1-8B

  Or use the Python SDK:
    python examples/quickstart.py
"""

import httpx


def main():
    api_url = "http://localhost:8000"

    # Check available models
    resp = httpx.get(f"{api_url}/v1/models")
    print("Available models:", resp.json())

    # Send a chat completion request
    resp = httpx.post(
        f"{api_url}/v1/chat/completions",
        json={
            "model": "meta-llama/Llama-3.1-8B",
            "messages": [
                {"role": "user", "content": "What is BitTorrent?"}
            ],
            "max_tokens": 128,
            "temperature": 0.7,
        },
        timeout=120,
    )
    data = resp.json()
    print("\nResponse:", data["choices"][0]["message"]["content"])
    print(f"\nTokens used: {data['usage']['total_tokens']}")

    # Check credit balance
    resp = httpx.get(f"{api_url}/aitorrent/credits")
    print("\nCredits:", resp.json())


if __name__ == "__main__":
    main()
