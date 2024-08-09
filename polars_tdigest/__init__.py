from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars.type_aliases import IntoExpr


lib = Path(__file__).parent


def estimate_median(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="estimate_median",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )


def tdigest(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="tdigest",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
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


def tdigest_cast(expr: IntoExpr) -> pl.Expr:
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        function_name="tdigest_cast",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )
