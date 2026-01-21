"""
Logging utilities for VEGA-Verified.
Provides structured logging with experiment tracking support.
"""

import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps
import time


# Global logger instances
_loggers: Dict[str, logging.Logger] = {}


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs for experiment tracking."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry["data"] = record.extra_data
        
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "vega_verified",
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = False
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        structured: Use JSON structured logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, level.upper()))
        
        # Always use structured format for file logs
        file_formatter = StructuredFormatter()
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "vega_verified") -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


def log_with_data(logger: logging.Logger, level: str, message: str, data: Dict[str, Any]) -> None:
    """
    Log a message with additional structured data.
    
    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        data: Additional data to include
    """
    record = logger.makeRecord(
        logger.name,
        getattr(logging, level.upper()),
        "",
        0,
        message,
        (),
        None
    )
    record.extra_data = data
    logger.handle(record)


def timed(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Optional logger instance (uses default if not provided)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger()
            
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"Completed {func.__name__} in {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Failed {func.__name__} after {elapsed:.3f}s: {e}")
                raise
        
        return wrapper
    return decorator


class ExperimentLogger:
    """Logger for tracking experiment progress and results."""
    
    def __init__(self, experiment_name: str, output_dir: str = "results"):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            name=f"experiment.{experiment_name}",
            log_file=str(self.output_dir / f"{experiment_name}.log"),
            structured=True
        )
        
        self.start_time = datetime.utcnow()
        self.events: list = []
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an experiment event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": event_type,
            "data": data
        }
        self.events.append(event)
        log_with_data(self.logger, "INFO", f"Event: {event_type}", data)
    
    def log_metric(self, name: str, value: Any, step: Optional[int] = None) -> None:
        """Log a metric value."""
        metric_data = {"name": name, "value": value}
        if step is not None:
            metric_data["step"] = step
        self.log_event("metric", metric_data)
    
    def log_comparison(self, mode1: str, mode2: str, results: Dict[str, Any]) -> None:
        """Log comparison between two modes."""
        self.log_event("comparison", {
            "mode1": mode1,
            "mode2": mode2,
            "results": results
        })
    
    def save_summary(self) -> str:
        """Save experiment summary to file."""
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "events": self.events
        }
        
        summary_path = self.output_dir / f"{self.experiment_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return str(summary_path)
