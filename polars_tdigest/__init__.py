from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


lib = Path(__file__).parent


def tdigest(expr: IntoExpr, max_size: int = 100) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="tdigest",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"max_size": max_size},
    )


def estimate_quantile(expr: IntoExpr, quantile: float) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="estimate_quantile",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"quantile": quantile},
    )


def tdigest_cast(expr: IntoExpr, max_size: int = 100) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="tdigest_cast",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
        kwargs={"max_size": max_size},
    )


def merge_tdigests(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="merge_tdigests",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )
