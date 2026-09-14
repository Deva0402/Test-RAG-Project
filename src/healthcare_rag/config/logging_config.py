import sys
from pathlib import Path
from loguru import logger
from healthcare_rag.config.settings import get_settings


def setup_logging() -> None:
    """
    configure application logging.
    call this once at application startup in main.py
    """
    settings=get_settings()
    logger.remove()

    logger.add(
        sys.stdout,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green>|"
            "<level>{level: <8}</level>|"
            "<cyan>{name}</cyan>:<cyan>{line}</cyan>|"
            "<level>{message}</level>"
        ),
        level=settings.log_level,
        colorize=True,
    )

    logs_dir =Path(settings.logs_dir)
    logs_dir.mkdir(parents=True,exist_ok=True)

    logger.add(
        str(logs_dir/"healthcare_rag_{time: YYYY-MM-DD}.log"),
        format="{time}|{level}|{name}:{line} | {message}",
        level=settings.log_level,
        rotation="00:00",
        retention="30 days",
        serialize=True,
        enqueue=True,
    )
    logger.info(
        f"logging ready|"
        f"env={settings.app_env.value}|"
        f"level={settings.log_level}"
    )
def get_logger(name: str):
    """get a named logger for any module.
    usage:
    from healthcare_rag.config.logging_config import get_logger
    log=get_logger(__name__)
    log.info(processing query)
    log.error("something failed!")"""
    return logger.bind(module=name)    
