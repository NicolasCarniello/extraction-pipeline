"""Mini-script de prueba asíncrono del pipeline de extracción.

Uso:
    python main.py                          # corre los 3 ejemplos incluidos
    python main.py --texto "Tu texto..."    # procesa un texto propio
    python main.py --provider anthropic     # fuerza un proveedor
"""

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

EJEMPLOS = {
    "Descripción de arquitectura": (
        "Nuestra API está construida con FastAPI y corre en contenedores Docker sobre Kubernetes. "
        "Usamos Redis como caché de sesiones y PostgreSQL para la persistencia. Desde que lanzamos la "
        "campaña de Black Friday, el pool de conexiones a PostgreSQL se agota con más de 500 usuarios "
        "concurrentes y los tiempos de respuesta pasan de 80 ms a 4 segundos."
    ),
    "Log de error en producción": (
        "2026-11-10T03:12:44Z ERROR [payment-service] Unhandled exception in /api/v2/checkout\n"
        "psycopg2.OperationalError: could not connect to server: Connection refused\n"
        "  Is the server running on host 'db-primary.internal' (10.0.3.12) and accepting TCP/IP "
        "connections on port 5432?\n"
        "2026-11-10T03:12:44Z WARN  [nginx] upstream timed out (110: Connection timed out) while "
        "reading response header from upstream\n"
        "2026-11-10T03:12:45Z ERROR [payment-service] 1.284 pagos rechazados en los últimos 5 minutos"
    ),
    "Prueba de estrés: texto ambiguo": (
        "Desde ayer el sistema anda raro. A veces carga y a veces no, y algunos usuarios se quejan. "
        "Ya reiniciamos lo de siempre pero sigue igual."
    ),
}


async def run(textos: dict[str, str]) -> int:
    # Import diferido: los logs ya están configurados cuando se crea la cadena.
    from chain import ExtractionError, process_text

    fallos = 0
    for titulo, texto in textos.items():
        print(f"\n{'═' * 70}\n  {titulo}\n{'═' * 70}")
        print(f"{texto}\n{'─' * 70}")
        try:
            resultado = await process_text(texto)
        except ExtractionError as exc:  # error controlado: el programa sigue con el próximo texto
            fallos += 1
            print(f"\n❌ Error controlado: {exc}")
            continue
        print("\n✅ Objeto validado (ExtraccionTecnica):")
        print(resultado.model_dump_json(indent=2))
    return 1 if fallos == len(textos) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline de extracción de entidades técnicas.")
    parser.add_argument("--texto", help="Texto a procesar (si se omite, se usan los ejemplos).")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="Sobrescribe LLM_PROVIDER.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Muestra también los logs HTTP.")
    args = parser.parse_args()

    load_dotenv()
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(name)s: %(message)s")
    if not args.verbose:  # los logs de cada request HTTP tapan los del pipeline
        logging.getLogger("httpx2").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    from chain import build_model

    try:
        build_model()  # valida la configuración antes de empezar
    except ValueError as exc:
        print(f"⚠️  {exc}")
        return 1

    textos = {"Texto ingresado": args.texto} if args.texto else EJEMPLOS
    return asyncio.run(run(textos))


if __name__ == "__main__":
    sys.exit(main())
