#![allow(clippy::unused_unit)]
use polars::prelude::*;

use crate::tdigest::{codecs::parse_tdigests, codecs::tdigest_to_series, TDigest};

use polars_core::export::rayon::prelude::*;
use polars_core::utils::arrow::array::Array;
use polars_core::utils::arrow::array::{Float32Array, Float64Array};
use polars_core::utils::arrow::array::{Int32Array, Int64Array};
use polars_core::POOL;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

// TODO: get rid of serde completely
#[derive(Debug, Deserialize)]
struct MergeTDKwargs {
    quantile: f64,
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
    let series = &inputs[0];
    // TODO: pooling is not feasible on small datasets
    let chunks = match series.dtype() {
        DataType::Float64 => {
            let values = series.f64()?;
            let chunks: Vec<TDigest> = POOL.install(|| {
                values
                    .downcast_iter()
                    .par_bridge()
                    .map(|chunk| {
                        let t = TDigest::new_with_size(kwargs.max_size);
                        let array = chunk.as_any().downcast_ref::<Float64Array>().unwrap();
                        let val_vec: Vec<f64> = array.non_null_values_iter().collect();
                        t.merge_unsorted(val_vec.to_owned())
                    })
                    .collect::<Vec<TDigest>>()
            });
            chunks
        }
        DataType::Float32 => {
            let values = series.f32()?;
            let chunks: Vec<TDigest> = POOL.install(|| {
                values
                    .downcast_iter()
                    .par_bridge()
                    .map(|chunk| {
                        let t = TDigest::new_with_size(kwargs.max_size);
                        let array = chunk.as_any().downcast_ref::<Float32Array>().unwrap();
                        let val_vec: Vec<f64> =
                            array.non_null_values_iter().map(|v| (v as f64)).collect();
                        t.merge_unsorted(val_vec.to_owned())
                    })
                    .collect::<Vec<TDigest>>()
            });
            chunks
        }
        DataType::Int64 => {
            let values = series.i64()?;
            let chunks: Vec<TDigest> = POOL.install(|| {
                values
                    .downcast_iter()
                    .par_bridge()
                    .map(|chunk| {
                        let t = TDigest::new_with_size(kwargs.max_size);
                        let array = chunk.as_any().downcast_ref::<Int64Array>().unwrap();
                        let val_vec: Vec<f64> =
                            array.non_null_values_iter().map(|v| (v as f64)).collect();
                        t.merge_unsorted(val_vec.to_owned())
                    })
                    .collect::<Vec<TDigest>>()
            });
            chunks
        }
        DataType::Int32 => {
            let values = series.i32()?;
            let chunks: Vec<TDigest> = POOL.install(|| {
                values
                    .downcast_iter()
                    .par_bridge()
                    .map(|chunk| {
                        let t = TDigest::new_with_size(kwargs.max_size);
                        let array = chunk.as_any().downcast_ref::<Int32Array>().unwrap();
                        let val_vec: Vec<f64> =
                            array.non_null_values_iter().map(|v| (v as f64)).collect();
                        t.merge_unsorted(val_vec.to_owned())
                    })
                    .collect::<Vec<TDigest>>()
            });
            chunks
        }
        _ => polars_bail!(InvalidOperation: "only supported for numerical types"),
    };

    let mut td_global = TDigest::merge_digests(chunks);
    if td_global.is_empty() {
        // Default value for TDigest contains NaNs that cause problems during serialization/deserailization
        td_global = TDigest::new(Vec::new(), 100.0, 0.0, 0.0, 0.0, 0)
    }
    Ok(tdigest_to_series(td_global, series.name()))
}

#[polars_expr(output_type_func=tdigest_output)]
fn tdigest_cast(inputs: &[Series], kwargs: TDigestKwargs) -> PolarsResult<Series> {
    let supported_dtypes = &[
        DataType::Float64,
        DataType::Float32,
        DataType::Int64,
        DataType::Int32,
    ];
    let series: Series = if supported_dtypes.contains(inputs[0].dtype()) {
        inputs[0].cast(&DataType::Float64)?
    } else {
        polars_bail!(InvalidOperation: "only supported for numerical types");
    };
    let values = series.f64()?;

    let chunks: Vec<TDigest> = POOL.install(|| {
        values
            .downcast_iter()
            .par_bridge()
            .map(|chunk| {
                let t = TDigest::new_with_size(kwargs.max_size);
                let array = chunk.as_any().downcast_ref::<Float64Array>().unwrap();
                t.merge_unsorted(array.values().to_vec())
            })
            .collect::<Vec<TDigest>>()
    });

    let t_global = TDigest::merge_digests(chunks);
    Ok(tdigest_to_series(t_global, series.name()))
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
fn estimate_quantile(inputs: &[Series], kwargs: MergeTDKwargs) -> PolarsResult<Series> {
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
