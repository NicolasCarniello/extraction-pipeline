"""Contrato de datos del pipeline: lo que el LLM DEBE devolver.

`with_structured_output(ExtraccionTecnica)` convierte esta clase en un JSON Schema
que se envía al modelo, y después valida la respuesta contra ella. Las reglas se
aplican en dos niveles:

1. En el JSON Schema (las respeta el propio modelo): tipos, campos obligatorios,
   valores del enum, lista con al menos un elemento.
2. En validadores de Pydantic (se aplican al recibir la respuesta): limpieza y
   deduplicación de tecnologías, longitud del resumen. Estas reglas viven en
   Python y no en el schema porque no todos los proveedores aceptan esas
   restricciones en modo estricto, y un schema rechazado sería un error 400.

Si la respuesta no cumple el contrato se lanza ValidationError, y la cadena reintenta.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

RESUMEN_MIN_CHARS = 20
RESUMEN_MAX_CHARS = 400

# Valores "de relleno" que un modelo puede usar para cumplir con una lista no vacía
# cuando el texto no menciona tecnologías. Aceptarlos sería validar datos falsos.
PLACEHOLDERS = {
    "n/a", "na", "ninguna", "ninguno", "desconocido", "desconocida", "no especificado",
    "no especificada", "no mencionado", "no mencionada", "sistema", "none", "unknown", "-",
}


class NivelCriticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto: arquitectura de software, log de error o incidente."""

    model_config = ConfigDict(extra="forbid")  # campos de más = respuesta inválida

    tecnologias: list[str] = Field(
        min_length=1,
        description=(
            "Tecnologías concretas mencionadas en el texto: lenguajes, frameworks, bases de datos, "
            "servicios cloud, herramientas o protocolos. Usar el nombre oficial (p. ej. 'PostgreSQL', "
            "no 'postgres'). Sin duplicados. No inventar tecnologías que el texto no mencione."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description=(
            "Impacto técnico. 'alta': caída del servicio, pérdida o corrupción de datos, brecha de "
            "seguridad o error en producción que afecta a usuarios. 'media': degradación de rendimiento, "
            "errores parciales o deuda técnica con riesgo concreto. 'baja': descripción informativa, "
            "advertencias sin impacto o mejoras opcionales."
        ),
    )
    resumen_tecnico: str = Field(
        description=(
            f"Resumen técnico en español de una o dos oraciones ({RESUMEN_MIN_CHARS} a "
            f"{RESUMEN_MAX_CHARS} caracteres): qué componentes intervienen y cuál es el problema o "
            "el punto clave de la arquitectura."
        ),
    )

    @field_validator("tecnologias")
    @classmethod
    def limpiar_tecnologias(cls, valores: list[str]) -> list[str]:
        """Quita espacios, vacíos, valores de relleno y duplicados (sin distinguir mayúsculas)."""
        vistas: set[str] = set()
        limpias: list[str] = []
        for valor in (v.strip() for v in valores):
            clave = valor.casefold()
            if valor and clave not in vistas and clave not in PLACEHOLDERS:
                vistas.add(clave)
                limpias.append(valor)
        if not limpias:
            raise ValueError(
                "la lista de tecnologías no puede estar vacía ni contener solo valores de relleno: "
                "el texto debe mencionar al menos una tecnología concreta"
            )
        return limpias

    @field_validator("resumen_tecnico")
    @classmethod
    def validar_resumen(cls, valor: str) -> str:
        valor = " ".join(valor.split())  # normaliza espacios y saltos de línea
        if not RESUMEN_MIN_CHARS <= len(valor) <= RESUMEN_MAX_CHARS:
            raise ValueError(
                f"el resumen debe tener entre {RESUMEN_MIN_CHARS} y {RESUMEN_MAX_CHARS} caracteres "
                f"(tiene {len(valor)})"
            )
        return valor
