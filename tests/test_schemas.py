import pytest
from pydantic import ValidationError

from schemas import ExtraccionTecnica, NivelCriticidad

VALIDO = {
    "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
    "nivel_de_criticidad": "alta",
    "resumen_tecnico": "API con caché en Redis y persistencia en PostgreSQL; cuello de botella en conexiones.",
}


def test_valid_object():
    obj = ExtraccionTecnica.model_validate(VALIDO)
    assert obj.nivel_de_criticidad is NivelCriticidad.ALTA
    assert obj.tecnologias == ["FastAPI", "Redis", "PostgreSQL"]


def test_serializes_to_expected_json():
    data = ExtraccionTecnica.model_validate(VALIDO).model_dump(mode="json")
    assert data == VALIDO


class TestTecnologias:
    def test_removes_duplicates_and_blanks_keeping_order(self):
        obj = ExtraccionTecnica.model_validate({**VALIDO, "tecnologias": [" Redis", "redis", "", "Kafka ", "REDIS"]})
        assert obj.tecnologias == ["Redis", "Kafka"]

    @pytest.mark.parametrize("tecnologias", [[], ["", "  "]])
    def test_rejects_empty_list(self, tecnologias):
        with pytest.raises(ValidationError, match="tecnologias"):
            ExtraccionTecnica.model_validate({**VALIDO, "tecnologias": tecnologias})

    @pytest.mark.parametrize("relleno", [["No especificado"], ["N/A", "desconocido"], ["Sistema"]])
    def test_rejects_placeholder_values(self, relleno):
        """Un modelo presionado a llenar la lista podría inventar valores: no los aceptamos."""
        with pytest.raises(ValidationError, match="relleno"):
            ExtraccionTecnica.model_validate({**VALIDO, "tecnologias": relleno})

    def test_placeholders_are_dropped_when_there_are_real_values(self):
        obj = ExtraccionTecnica.model_validate({**VALIDO, "tecnologias": ["Redis", "N/A"]})
        assert obj.tecnologias == ["Redis"]

    def test_rejects_non_list(self):
        with pytest.raises(ValidationError):
            ExtraccionTecnica.model_validate({**VALIDO, "tecnologias": "FastAPI, Redis"})


class TestCriticidad:
    @pytest.mark.parametrize("nivel", ["baja", "media", "alta"])
    def test_accepts_enum_values(self, nivel):
        assert ExtraccionTecnica.model_validate({**VALIDO, "nivel_de_criticidad": nivel}).nivel_de_criticidad == nivel

    @pytest.mark.parametrize("nivel", ["critica", "ALTA ", "high", ""])
    def test_rejects_other_values(self, nivel):
        with pytest.raises(ValidationError, match="nivel_de_criticidad"):
            ExtraccionTecnica.model_validate({**VALIDO, "nivel_de_criticidad": nivel})


class TestResumen:
    def test_normalizes_whitespace(self):
        obj = ExtraccionTecnica.model_validate({**VALIDO, "resumen_tecnico": "  API   con\ncaché  en Redis y Postgres.  "})
        assert obj.resumen_tecnico == "API con caché en Redis y Postgres."

    @pytest.mark.parametrize("resumen", ["corto", "x" * 401])
    def test_rejects_bad_length(self, resumen):
        with pytest.raises(ValidationError, match="resumen_tecnico"):
            ExtraccionTecnica.model_validate({**VALIDO, "resumen_tecnico": resumen})


def test_rejects_missing_fields():
    with pytest.raises(ValidationError) as exc_info:
        ExtraccionTecnica.model_validate({"tecnologias": ["Redis"]})
    faltantes = {e["loc"][0] for e in exc_info.value.errors()}
    assert faltantes == {"nivel_de_criticidad", "resumen_tecnico"}


def test_rejects_extra_fields():
    with pytest.raises(ValidationError, match="extra"):
        ExtraccionTecnica.model_validate({**VALIDO, "comentario": "no pedido"})


def test_every_field_has_a_description():
    """Las descripciones viajan al LLM dentro del JSON Schema: son parte del prompt."""
    assert all(field.description for field in ExtraccionTecnica.model_fields.values())
