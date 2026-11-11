"""Tests de integración: ChatOpenAI / ChatAnthropic REALES contra un servidor HTTP simulado.

El modelo falso de test_chain.py imita a LangChain; estos tests verifican que LangChain
y los SDKs se comporten de verdad así (formato de la request, parseo y errores), sin
red ni API keys. Así se descubrió, por ejemplo, que el SDK de OpenAI lanza
LengthFinishReasonError por su cuenta cuando la respuesta se corta.
"""

import json
import logging

import anthropic
import httpx2
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from chain import ExtractionError, build_chain, process_text

OK = json.dumps({
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API en FastAPI con caché Redis; el pool de PostgreSQL se agota con alta concurrencia.",
})
SIN_TECNOLOGIAS = json.dumps({
    "tecnologias": [],
    "nivel_de_criticidad": "media",
    "resumen_tecnico": "El sistema falla de forma intermitente sin causa clara.",
})
TEXTO = "FastAPI + Redis + PostgreSQL: el pool de conexiones se agota con 500 usuarios."


class FakeServer:
    def __init__(self, *responses: dict):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        return httpx2.Response(200, json=self.responses.pop(0))

    def client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self.handler))


# ---------------------------------------------------------------- Anthropic
def anthropic_message(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 40},
    }


def anthropic_chain(server: FakeServer):
    model = ChatAnthropic(model="claude-opus-5-5", api_key="sk-ant-test", max_tokens=4096, max_retries=0)
    # ChatAnthropic no expone un parámetro de cliente HTTP: reemplazamos el cliente async cacheado.
    model.__dict__["_async_client"] = anthropic.AsyncAnthropic(
        api_key="sk-ant-test", http_client=server.client(), max_retries=0
    )
    return build_chain(model, esperar_entre_intentos=False)


class TestAnthropic:
    async def test_request_uses_native_structured_output(self):
        server = FakeServer(anthropic_message(OK))
        resultado = await process_text(TEXTO, chain=anthropic_chain(server))
        assert resultado.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
        body = server.requests[0]
        assert body["output_config"]["format"]["type"] == "json_schema"
        assert "tool_choice" not in body and "temperature" not in body
        assert body["system"].startswith("Sos un ingeniero")  # el rol system va como parámetro aparte

    async def test_retries_on_schema_violation(self):
        server = FakeServer(anthropic_message(SIN_TECNOLOGIAS), anthropic_message(OK))
        assert (await process_text(TEXTO, chain=anthropic_chain(server))).tecnologias
        assert len(server.requests) == 2

    async def test_retries_on_truncated_response(self, caplog):
        caplog.set_level(logging.INFO, logger="pipeline")
        server = FakeServer(anthropic_message(OK[:40], stop_reason="max_tokens"), anthropic_message(OK))
        await process_text(TEXTO, chain=anthropic_chain(server))
        assert len(server.requests) == 2
        assert "Respuesta incompleta" in caplog.text

    async def test_refusal_is_not_retried(self):
        server = FakeServer(anthropic_message("", stop_reason="refusal"))
        with pytest.raises(ExtractionError, match="negó"):
            await process_text(TEXTO, chain=anthropic_chain(server))
        assert len(server.requests) == 1


# ---------------------------------------------------------------- OpenAI
def openai_completion(text: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "finish_reason": finish_reason,
                     "message": {"role": "assistant", "content": text, "refusal": None}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 40, "total_tokens": 90},
    }


def openai_chain(server: FakeServer):
    model = ChatOpenAI(
        model="gpt-4o-mini", api_key="sk-test", temperature=0, max_retries=0, http_async_client=server.client()
    )
    return build_chain(model, esperar_entre_intentos=False)


class TestOpenAI:
    async def test_request_uses_strict_json_schema(self):
        server = FakeServer(openai_completion(OK))
        resultado = await process_text(TEXTO, chain=openai_chain(server))
        assert resultado.nivel_de_criticidad == "alta"
        body = server.requests[0]
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["strict"] is True
        assert body["temperature"] == 0
        assert [m["role"] for m in body["messages"]] == ["system", "user"]

    async def test_retries_on_schema_violation(self):
        server = FakeServer(openai_completion(SIN_TECNOLOGIAS), openai_completion(OK))
        assert (await process_text(TEXTO, chain=openai_chain(server))).tecnologias
        assert len(server.requests) == 2

    async def test_retries_when_sdk_raises_length_error(self, caplog):
        caplog.set_level(logging.INFO, logger="pipeline")
        server = FakeServer(openai_completion(OK[:40], finish_reason="length"), openai_completion(OK))
        await process_text(TEXTO, chain=openai_chain(server))
        assert len(server.requests) == 2
        assert "LengthFinishReasonError" in caplog.text

    async def test_self_correction_sends_error_back_to_the_model(self):
        server = FakeServer(*[openai_completion(SIN_TECNOLOGIAS)] * 3, openai_completion(OK))
        assert (await process_text(TEXTO, chain=openai_chain(server))).tecnologias
        assert len(server.requests) == 4
        ultimo = server.requests[-1]["messages"][-1]
        assert ultimo["role"] == "user" and "no pasó la validación" in ultimo["content"]


async def test_self_correction_prompt_states_the_concrete_cause():
    """Con Anthropic el ValidationError llega envuelto en OutputParserException: el prompt
    de auto-corrección debe mostrar la causa concreta, no el JSON completo."""
    server = FakeServer(*[anthropic_message(SIN_TECNOLOGIAS)] * 3, anthropic_message(OK))
    await process_text(TEXTO, chain=anthropic_chain(server))
    ultimo = server.requests[-1]["messages"][-1]["content"]
    texto = ultimo if isinstance(ultimo, str) else " ".join(b["text"] for b in ultimo)
    assert "tecnologias: List should have at least 1 item" in texto
    assert "Failed to parse" not in texto
