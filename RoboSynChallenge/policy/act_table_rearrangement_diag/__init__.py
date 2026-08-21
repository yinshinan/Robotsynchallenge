"""Non-destructive ACT diagnostics for table_rearrangement."""

# Importing the task registers the diagnostic Gym environment before eval creates it.
from .strict_task import TableRearrangementDiagnosticEnv
from .deploy_policy import eval, get_model, reset_model

__all__ = [
    "TableRearrangementDiagnosticEnv",
    "eval",
    "get_model",
    "reset_model",
]
