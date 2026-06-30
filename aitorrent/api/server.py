from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="AITorrent", version="0.1.0")

# These get set by the CLI when the server starts
_pipeline = None
_manifest = None
_ledger = None
_tokenizer = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, alias="max_tokens")
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


def configure(pipeline, manifest, ledger, tokenizer):
    global _pipeline, _manifest, _ledger, _tokenizer
    _pipeline = pipeline
    _manifest = manifest
    _ledger = ledger
    _tokenizer = tokenizer


@app.get("/v1/models")
async def list_models():
    models = []
    if _manifest:
        models.append({
            "id": _manifest.model_id,
            "object": "model",
            "owned_by": "aitorrent",
        })
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if _pipeline is None or _tokenizer is None:
        raise HTTPException(503, "Pipeline not initialized")

    prompt = _format_messages(request.messages)
    input_ids = _tokenizer.encode(prompt, return_tensors="pt")[0]

    if request.stream:
        return StreamingResponse(
            _stream_response(request, input_ids),
            media_type="text/event-stream",
        )

    generated = await _pipeline.generate(
        input_ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )

    output_text = _tokenizer.decode(generated[len(input_ids):], skip_special_tokens=True)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": output_text},
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": len(input_ids),
            "completion_tokens": len(generated) - len(input_ids),
            "total_tokens": len(generated),
        },
    )


async def _stream_response(
    request: ChatCompletionRequest, input_ids: torch.Tensor
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    async for token_id in _pipeline.generate_stream(
        input_ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    ):
        text = _tokenizer.decode([token_id], skip_special_tokens=True)
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "delta": {"content": text},
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/aitorrent/credits")
async def get_credits():
    if _ledger is None:
        raise HTTPException(503, "Ledger not initialized")
    return {
        "balance": _ledger.balance,
        "total_earned": _ledger.total_earned(),
        "total_spent": _ledger.total_spent(),
    }


def _format_messages(messages: list[ChatMessage]) -> str:
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)
