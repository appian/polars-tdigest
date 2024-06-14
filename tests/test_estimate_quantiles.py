import polars as pl
import pytest
from polars import Float32, Int32, Utf8, col

from polars_tdigest import estimate_median, estimate_quantile, tdigest

df_int = pl.DataFrame(
    {
        "values": [1, 3, 2, 5, 7],
        "group": ["a", "a", "a", "b", "b"],
    }
)

df_float = pl.DataFrame(
    {
        "values": [1.0, 3.0, 2.0, 5.5, 7.4],
        "group": ["a", "a", "a", "b", "b"],
    }
)


def test_estimate_median():

    median_df = (
        df_int.group_by("group")
        .agg(
            [
                estimate_median("values").alias("estimate_median"),
                pl.col("values").median().alias("median"),
            ]
        )
        .select(
            [
                "estimate_median",
                "median",
                (col("estimate_median") - col("median")).alias("diff"),
            ]
        )
    )

    assert median_df.filter(col("diff") != 0.0).height == 0


def test_estimate_quantile_int64():
    df_median = (
        df_int.with_columns()
        .group_by("group")
        .agg(
            [
                tdigest("values").alias("metric-0"),
            ]
        )
    )
    df_merged_median = df_median.select(estimate_quantile("metric-0", 0.5))

    assert df_merged_median.item() == 3.0


def test_estimate_quantile_int32():
    df_median = (
        df_int.with_columns(col("values").cast(Int32))
        .group_by("group")
        .agg(
            [
                tdigest("values").alias("metric-0"),
            ]
        )
    )
    df_merged_median = df_median.select(estimate_quantile("metric-0", 0.5))

    assert df_merged_median.item() == 3.0


def test_estimate_quantile_f64():
    df_median = df_float.group_by("group").agg(
        [
            tdigest("values").alias("metric-0"),
        ]
    )
    df_merged_median = df_median.select(estimate_quantile("metric-0", 0.5))

    assert df_merged_median.item() == 3.0


def test_estimate_quantile_f32():

    df_median = (
        df_float.with_columns(col("values").cast(Float32))
        .group_by("group")
        .agg(
            [
                tdigest("values").alias("metric-0"),
            ]
        )
    )
    df_merged_median = df_median.select(estimate_quantile("metric-0", 0.5))

    assert df_merged_median.item() == 3.0


def test_estimate_quantile_utf8():

    with pytest.raises(pl.exceptions.ComputeError):
        (
            df_float.with_columns(col("values").cast(Utf8))
            .group_by("group")
            .agg(
                [
                    tdigest("values").alias("metric-0"),
                ]
            )
        )
