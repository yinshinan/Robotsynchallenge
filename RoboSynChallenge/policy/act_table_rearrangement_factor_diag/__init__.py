"""Extended, isolated ACT diagnostics for table_rearrangement factors."""

from policy.act_table_rearrangement_diag.strict_task import (
    TableRearrangementDiagnosticEnv,
)

from .deploy_policy import eval, get_model, reset_model

__all__ = [
    "TableRearrangementDiagnosticEnv",
    "eval",
    "get_model",
    "reset_model",
]
