"""Interactive setup wizard for ``plugmem init``."""
from plugmem.cli.wizard.main import run_wizard
from plugmem.cli.wizard.sections import (
    run_embedding_section,
    run_llm_section,
    run_service_section,
)

__all__ = [
    "run_wizard",
    "run_llm_section",
    "run_embedding_section",
    "run_service_section",
]
