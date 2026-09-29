"""
Structured Logging with Rich formatting.
"""

import logging
import sys
from typing import Optional
from rich.console import Console
from rich.logging import RichHandler

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)

def setup_logger(name: str = "buffer_automation", level: str = "INFO") -> logging.Logger:
    """Configures and returns a rich-formatted logger."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Avoid adding multiple handlers if already configured
    if not logger.handlers:
        handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            show_time=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

logger = setup_logger()
