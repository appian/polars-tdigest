#![allow(clippy::unused_unit)]
use polars::prelude::*;

use crate::tdigest::{Centroid, TDigest};
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

// TODO: error handling w/o panic
fn parse_tdigests(input: &Series) -> Vec<TDigest> {
    input
        .struct_()
        .into_iter()
        .flat_map(|chunk| {
            let count_series = chunk.field_by_name("count").unwrap();
            let count_it = count_series.i64().unwrap().into_iter();
            let max_series = chunk.field_by_name("max").unwrap();
            let min_series = chunk.field_by_name("min").unwrap();
            let sum_series = chunk.field_by_name("sum").unwrap();
            let max_size_series = chunk.field_by_name("max_size").unwrap();
            let centroids_series = chunk.field_by_name("centroids").unwrap();
            let mut max_it = max_series.f64().unwrap().into_iter();
            let mut min_it = min_series.f64().unwrap().into_iter();
            let mut max_size_it = max_size_series.i64().unwrap().into_iter();
            let mut sum_it = sum_series.f64().unwrap().into_iter();
            let mut centroids_it = centroids_series.list().unwrap().into_iter();

            count_it
                .map(|c| {
                    let centroids = centroids_it.next().unwrap().unwrap();
                    let mean_series = centroids.struct_().unwrap().field_by_name("mean").unwrap();
                    let mean_it = mean_series.f64().unwrap().into_iter();
                    let weight_series = centroids
                        .struct_()
                        .unwrap()
                        .field_by_name("weight")
                        .unwrap();
                    let mut weight_it = weight_series.i64().unwrap().into_iter();
                    let centroids_res = mean_it
                        .map(|m| {
                            Centroid::new(m.unwrap(), weight_it.next().unwrap().unwrap() as f64)
                        })
                        .collect::<Vec<_>>();
                    TDigest::new(
                        centroids_res,
                        sum_it.next().unwrap().unwrap(),
                        c.unwrap() as f64,
                        max_it.next().unwrap().unwrap(),
                        min_it.next().unwrap().unwrap(),
                        max_size_it.next().unwrap().unwrap() as usize,
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn tdigest_to_series(tdigest: TDigest, name: &str) -> Series {
    let mut means: Vec<f64> = vec![];
    let mut weights: Vec<i64> = vec![];
    tdigest.centroids().iter().for_each(|c| {
        weights.push(c.weight() as i64);
        means.push(c.mean());
    });

    let centroids_series = DataFrame::new(vec![
        Series::new("mean", means),
        Series::new("weight", weights),
    ])
    .unwrap()
    .into_struct("centroids")
    .into_series();

    DataFrame::new(vec![
        Series::new("centroids", [Series::new("centroids", centroids_series)]),
        Series::new("sum", [tdigest.sum()]),
        Series::new("min", [tdigest.min()]),
        Series::new("max", [tdigest.max()]),
        Series::new("count", [tdigest.count() as i64]),
        Series::new("max_size", [tdigest.max_size() as i64]),
    ])
    .unwrap()
    .into_struct(name)
    .into_series()
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    #[test]
    fn test_tdigest_deserializstion() {
        let json_str = "[{\"tdigest\":{\"centroids\":[{\"mean\":4.0,\"weight\":1},{\"mean\":5.0,\"weight\":1},{\"mean\":6.0,\"weight\":1}],\"sum\":15.0,\"min\":4.0,\"max\":6.0,\"count\":3,\"max_size\":100}},{\"tdigest\":{\"centroids\":[{\"mean\":1.0,\"weight\":1},{\"mean\":2.0,\"weight\":1},{\"mean\":3.0,\"weight\":1}],\"sum\":6.0,\"min\":1.0,\"max\":3.0,\"count\":3,\"max_size\":100}}]";
        let cursor = Cursor::new(json_str);
        let df = JsonReader::new(cursor).finish().unwrap();
        let series = df.column("tdigest").unwrap();
        let res = parse_tdigests(series);
        let expected = vec![
            TDigest::new(
                vec![
                    Centroid::new(4.0, 1.0),
                    Centroid::new(5.0, 1.0),
                    Centroid::new(6.0, 1.0),
                ],
                15.0,
                3.0,
                6.0,
                4.0,
                100,
            ),
            TDigest::new(
                vec![
                    Centroid::new(1.0, 1.0),
                    Centroid::new(2.0, 1.0),
                    Centroid::new(3.0, 1.0),
                ],
                6.0,
                3.0,
                3.0,
                1.0,
                100,
            ),
        ];
        assert!(res == expected);
    }

    #[test]
    fn test_tdigest_serialization() {
        let tdigest = TDigest::new(
            vec![
                Centroid::new(10.0, 1.0),
                Centroid::new(20.0, 2.0),
                Centroid::new(30.0, 3.0),
            ],
            60.0,
            3.0,
            30.0,
            10.0,
            300,
        );
        let res = tdigest_to_series(tdigest, "n");

        let cs = DataFrame::new(vec![
            Series::new("mean", [10.0, 20.0, 30.0]),
            Series::new("weight", [1, 2, 3]),
        ])
        .unwrap()
        .into_struct("centroids")
        .into_series();

        let expected = DataFrame::new(vec![
            Series::new("centroids", [Series::new("a", cs)]),
            Series::new("sum", [60.0]),
            Series::new("min", [10.0]),
            Series::new("max", [30.0]),
            Series::new("count", [3.0]),
            Series::new("max_size", [300 as i64]),
        ])
        .unwrap()
        .into_struct("n")
        .into_series();

        assert!(res == expected);
    }
}
