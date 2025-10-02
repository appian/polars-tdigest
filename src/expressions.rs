#![allow(clippy::unused_unit)]
use polars::prelude::*;

use crate::tdigest::{codecs::parse_tdigests, codecs::tdigest_to_series, TDigest};

use polars_core::export::rayon::prelude::*;
use polars_core::utils::arrow::array::Array;
use polars_core::utils::arrow::array::Float64Array;
use polars_core::POOL;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

static SUPPORTED_TYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Int64,
    DataType::Int32,
    DataType::UInt64,
    DataType::UInt32,
];

// TODO: get rid of serde completely
#[derive(Debug, Deserialize)]
struct QuantileKwargs {
    quantile: f64,
}

#[derive(Debug, Deserialize)]
struct CDFKwargs {
    x: f64,
}

#[derive(Debug, Deserialize)]
struct TDigestKwargs {
    max_size: usize,
}

fn tdigest_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new("tdigest", DataType::Struct(tdigest_fields())))
}

fn tdigest_fields() -> Vec<Field> {
    vec![
        Field::new(
            "centroids",
            DataType::List(Box::new(DataType::Struct(vec![
                Field::new("mean", DataType::Float64),
                Field::new("weight", DataType::Int64),
            ]))),
        ),
        Field::new("sum", DataType::Float64),
        Field::new("min", DataType::Float64),
        Field::new("max", DataType::Float64),
        Field::new("count", DataType::Int64),
        Field::new("max_size", DataType::Int64),
    ]
}

// Todo support other numerical types
#[polars_expr(output_type_func=tdigest_output)]
fn tdigest(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    let mut tdigest = tdigest_from_series(inputs, kwargs.max_size)?;
    if tdigest.is_empty() {
        // Default value for TDigest contains NaNs that cause problems during serialization/deserailization
        tdigest = TDigest::new(Vec::new(), 100.0, 0.0, 0.0, 0.0, 0)
    }
    Ok(tdigest_to_series(tdigest, inputs[0].name()))
}

#[polars_expr(output_type_func=tdigest_output)]
fn tdigest_cast(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    let tdigest = tdigest_from_series(inputs, kwargs.max_size)?;
    Ok(tdigest_to_series(tdigest, inputs[0].name()))
}

fn tdigest_from_series(inputs: &[Series], max_size: usize) -> PolarsResult<TDigest> {
    let series = &inputs[0];
    let series_casted: &Series = if series.dtype() == &DataType::Float64 {
        series
    } else {
        if !SUPPORTED_TYPES.contains(series.dtype()) {
            polars_bail!(InvalidOperation: "only supported for numerical types");
        }
        let cast_result = series.cast(&DataType::Float64);
        if cast_result.is_err() {
            polars_bail!(InvalidOperation: "only supported for numerical types");
        }
        &cast_result.unwrap()
    };

    let values = series_casted.f64()?;
    let chunks: Vec<TDigest> = POOL.install(|| {
        values
            .downcast_iter()
            .par_bridge()
            .map(|chunk| {
                let t = TDigest::new_with_size(max_size);
                let array = chunk.as_any().downcast_ref::<Float64Array>().unwrap();
                t.merge_unsorted(array.non_null_values_iter().collect())
            })
            .collect::<Vec<TDigest>>()
    });

    Ok(TDigest::merge_digests(chunks))
}

fn parse_tdigest(inputs: &[Series]) -> TDigest {
    let tdigests: Vec<TDigest> = parse_tdigests(&inputs[0]);
    TDigest::merge_digests(tdigests)
}

#[polars_expr(output_type_func=tdigest_output)]
fn merge_tdigests(inputs: &[Series]) -> PolarsResult<Series> {
    let tdigest = parse_tdigest(inputs);
    Ok(tdigest_to_series(tdigest, inputs[0].name()))
}

// TODO this should check the type of the series and also work on series of Type f64
#[polars_expr(output_type=Float64)]
fn estimate_quantile(inputs: &[Series], kwargs: QuantileKwargs) -> PolarsResult<Series> {
    let tdigest = parse_tdigest(inputs);
    if tdigest.is_empty() {
        let v: &[Option<f64>] = &[None];
        Ok(Series::new("", v))
    } else {
        let ans = tdigest.estimate_quantile(kwargs.quantile);
        Ok(Series::new("", vec![ans]))
    }
}

#[polars_expr(output_type=Float64)]
fn estimate_cdf(inputs: &[Series], kwargs: CDFKwargs) -> PolarsResult<Series> {
    let tdigest = parse_tdigest(inputs);
    if tdigest.is_empty() {
        let v: &[Option<f64>] = &[None];
        Ok(Series::new("", v))
    } else {
        let ans = tdigest.estimate_cdf(kwargs.x);
        Ok(Series::new("", vec![ans]))
    }
}

#[polars_expr(output_type=Float64)]
fn estimate_median(inputs: &[Series]) -> PolarsResult<Series> {
    let tdigest = parse_tdigest(inputs);
    if tdigest.is_empty() {
        let v: &[Option<f64>] = &[None];
        Ok(Series::new("", v))
    } else {
        let ans = tdigest.estimate_median();
        Ok(Series::new("", vec![ans]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supporting_different_numeric_types() {
        // These types are not supported w/o special compilation flags
        // The plugin panics if these types are used probably due to magic in #[palars_expr] annotations
        let unsupported_types = [
            DataType::Int16,
            DataType::Int8,
            DataType::UInt16,
            DataType::UInt8,
        ];

        SUPPORTED_TYPES.iter().for_each(|t| {
            let series = [Series::new("n", [1, 2, 3]).cast(t).unwrap()];
            let td = tdigest_from_series(&series, 200).unwrap();
            assert!(td.estimate_median() == 2.0);
            assert!(td.estimate_cdf(2.0) == 0.5);
            assert!((td.estimate_cdf(2.5) - 0.66666).abs() < 0.0001);
        });

        unsupported_types.iter().for_each(|t| {
            assert!(Series::new("n", [1, 2, 3]).cast(t).is_err());
        });
    }
}
