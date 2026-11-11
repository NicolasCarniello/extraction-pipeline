"""Tests del pipeline con un modelo falso: sin red, sin API keys y con fallos controlados."""

import json
import logging

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from chain import (
    ExtractionError,
    build_chain,
    build_model,
    build_prompt,
    process_text,
    validar_salida,
)
from schemas import ExtraccionTecnica

TEXTO = "La API en FastAPI usa Redis y PostgreSQL; el pool de conexiones se agota con 500 usuarios."
VALIDO = json.dumps({
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; se agota el pool de conexiones.",
})
SIN_TECNOLOGIAS = json.dumps({
    "tecnologias": [],
    "nivel_de_criticidad": "media",
    "resumen_tecnico": "El sistema presenta fallas intermitentes sin causa identificada.",
})
JSON_ROTO = '{"tecnologias": ["FastAPI", "Redis"], "nivel_de_criti'


class FakeStructuredModel:
    """Imita `model.with_structured_output(Schema, include_raw=True)`.

    Cada respuesta es (contenido, metadata). Devuelve {"raw", "parsed", "parsing_error"}
    igual que LangChain, o lanza la excepción si la respuesta es una excepción.
    """

    def __init__(self, *respuestas):
        self.respuestas = list(respuestas)
        self.prompts: list[list] = []

    @property
    def calls(self) -> int:
        return len(self.prompts)

    def with_structured_output(self, schema, *, include_raw=False, method=None):
        assert include_raw is True and method == "json_schema"

        def responder(prompt_value):
            self.prompts.append(prompt_value.to_messages())
            respuesta = self.respuestas.pop(0)
            if isinstance(respuesta, Exception):
                raise respuesta
            contenido, metadata = respuesta if isinstance(respuesta, tuple) else (respuesta, {"finish_reason": "stop"})
            raw = AIMessage(content=contenido, response_metadata=metadata)
            try:
                return {"raw": raw, "parsed": schema.model_validate_json(contenido), "parsing_error": None}
            except ValidationError as exc:
                return {"raw": raw, "parsed": None, "parsing_error": exc}

        return RunnableLambda(responder)


def chain_with(*respuestas):
    model = FakeStructuredModel(*respuestas)
    return model, build_chain(model, esperar_entre_intentos=False)


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="pipeline")
    return caplog


class TestPrompt:
    def test_only_input_variable_is_texto(self):
        assert build_prompt().input_variables == ["texto"]  # instrucciones_formato ya viene cargada

    def test_has_system_and_human_roles(self):
        messages = build_prompt().format_messages(texto=TEXTO)
        assert [m.type for m in messages] == ["system", "human"]
        assert TEXTO in messages[1].content

    def test_format_instructions_come_from_the_schema(self):
        system = build_prompt().format_messages(texto=TEXTO)[0].content
        for campo in ExtraccionTecnica.model_fields:
            assert f"- {campo}:" in system


class TestHappyPath:
    async def test_returns_validated_object(self, logs):
        model, chain = chain_with(VALIDO)
        resultado = await process_text(TEXTO, chain=chain)
        assert isinstance(resultado, ExtraccionTecnica)
        assert resultado.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
        assert model.calls == 1
        assert "✓ Validación correcta" in logs.text

    async def test_text_reaches_the_model(self):
        model, chain = chain_with(VALIDO)
        await process_text(f"  {TEXTO}  ", chain=chain)
        assert TEXTO in model.prompts[0][-1].content

    async def test_empty_text_is_rejected_before_calling_the_model(self):
        model, chain = chain_with(VALIDO)
        with pytest.raises(ValueError, match="vacío"):
            await process_text("   ", chain=chain)
        assert model.calls == 0


class TestRetry:
    async def test_recovers_from_malformed_json(self, logs):
        model, chain = chain_with(JSON_ROTO, VALIDO)
        resultado = await process_text(TEXTO, chain=chain)
        assert resultado.nivel_de_criticidad == "alta"
        assert model.calls == 2
        assert "✗ Validación fallida" in logs.text
        assert "🔁 Reintento automático: intento 2 de 3" in logs.text

    async def test_recovers_from_schema_violation(self):
        model, chain = chain_with(SIN_TECNOLOGIAS, VALIDO)
        assert (await process_text(TEXTO, chain=chain)).tecnologias
        assert model.calls == 2

    @pytest.mark.parametrize(
        "metadata",
        [{"finish_reason": "length"}, {"stop_reason": "max_tokens"}],
        ids=["openai-length", "anthropic-max_tokens"],
    )
    async def test_detects_truncated_response_even_if_json_parses(self, logs, metadata):
        """Un JSON que parsea pero vino cortado por tokens NO se acepta."""
        model, chain = chain_with((VALIDO, metadata), VALIDO)
        await process_text(TEXTO, chain=chain)
        assert model.calls == 2
        assert "Respuesta incompleta" in logs.text

    async def test_parser_exception_is_retried(self):
        model, chain = chain_with(OutputParserException("sin tool call"), VALIDO)
        await process_text(TEXTO, chain=chain)
        assert model.calls == 2


class TestSelfCorrection:
    async def test_fallback_shows_the_validation_error_to_the_model(self, logs):
        model, chain = chain_with(SIN_TECNOLOGIAS, SIN_TECNOLOGIAS, SIN_TECNOLOGIAS, VALIDO)
        resultado = await process_text(TEXTO, chain=chain)
        assert resultado.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]
        assert model.calls == 4  # 3 intentos + 1 auto-corrección
        ultimo_mensaje = model.prompts[-1][-1].content
        assert "no pasó la validación" in ultimo_mensaje
        assert "tecnologias" in ultimo_mensaje  # el error concreto llega al modelo
        assert "🩹" in logs.text

    async def test_gives_up_with_controlled_error(self, logs):
        model, chain = chain_with(*[SIN_TECNOLOGIAS] * 4)
        with pytest.raises(ExtractionError, match="tecnologias"):
            await process_text(TEXTO, chain=chain)
        assert model.calls == 4
        assert "Extracción fallida" in logs.text


class TestNonRetryable:
    async def test_refusal_is_not_retried(self):
        model, chain = chain_with(("", {"stop_reason": "refusal"}))
        with pytest.raises(ExtractionError, match="negó"):
            await process_text(TEXTO, chain=chain)
        assert model.calls == 1

    async def test_api_errors_are_wrapped(self):
        model, chain = chain_with(RuntimeError("401 invalid x-api-key"))
        with pytest.raises(ExtractionError, match="API"):
            await process_text(TEXTO, chain=chain)
        assert model.calls == 1  # los errores de API los reintenta el SDK, no with_retry


def test_validar_salida_accepts_dict_from_provider():
    raw = AIMessage(content="", response_metadata={"finish_reason": "stop"})
    parsed = json.loads(VALIDO)
    assert isinstance(validar_salida({"raw": raw, "parsed": parsed, "parsing_error": None}), ExtraccionTecnica)


# ---------------------------------------------------------------- proveedores reales (sin red)
@pytest.fixture
def env(monkeypatch):
    for var in ["LLM_PROVIDER", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_MODEL", "ANTHROPIC_MODEL",
                "LLM_TEMPERATURE", "LLM_MAX_TOKENS"]:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def _bound_kwargs(structured) -> dict:
    """Busca los kwargs que with_structured_output le agrega a la request del modelo."""
    from langchain_core.runnables import RunnableBinding

    pendientes = [structured]
    while pendientes:
        r = pendientes.pop()
        if isinstance(r, RunnableBinding) and r.kwargs:
            return r.kwargs
        for attr in ("first", "middle", "last", "bound", "runnable"):
            valor = getattr(r, attr, None)
            if valor is not None:
                pendientes.extend(valor if isinstance(valor, list) else [valor])
        if hasattr(r, "steps__"):
            pendientes.extend(r.steps__.values())
    raise AssertionError("no se encontraron kwargs")


class TestProviders:
    def test_openai_uses_strict_schema_compatible_with_the_api(self, env):
        from openai.lib._pydantic import to_strict_json_schema

        env.setenv("OPENAI_API_KEY", "sk-test")
        model = build_model("openai")
        assert model.temperature == 0
        kwargs = _bound_kwargs(model.with_structured_output(ExtraccionTecnica, include_raw=True, method="json_schema"))
        assert kwargs["response_format"] is ExtraccionTecnica
        schema = json.dumps(to_strict_json_schema(ExtraccionTecnica))
        # El modo estricto de OpenAI no admite minLength/maxLength: esas reglas viven en los validadores.
        assert "minLength" not in schema and "maxLength" not in schema

    def test_anthropic_uses_native_json_schema_without_forced_tools(self, env):
        env.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        env.setenv("LLM_TEMPERATURE", "1.5")
        model = build_model("anthropic")
        kwargs = _bound_kwargs(model.with_structured_output(ExtraccionTecnica, include_raw=True, method="json_schema"))
        assert kwargs["output_config"]["format"]["type"] == "json_schema"
        assert "tool_choice" not in kwargs and "tools" not in kwargs
        payload = model._get_request_payload([("human", "hola")])
        assert "temperature" not in payload and payload["max_tokens"] == 4096

    def test_missing_key(self, env):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            build_model()

    def test_unknown_provider(self, env):
        with pytest.raises(ValueError, match="Proveedor desconocido"):
            build_model("gemini")
