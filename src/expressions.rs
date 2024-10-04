#![allow(clippy::unused_unit)]
use polars::prelude::*;

use polars_core::export::rayon::prelude::*;
use polars_core::utils::arrow::array::Array;
use polars_core::utils::arrow::array::{Float32Array, Float64Array};
use polars_core::utils::arrow::array::{Int32Array, Int64Array};
use polars_core::POOL;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::fmt::Write;
use std::io::BufWriter;
use std::io::Cursor;
use std::num::NonZeroUsize;
use tdigest::TDigest;

// mod tdigest;

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca: &StringChunked = inputs[0].str()?;
    let out: StringChunked = ca.apply_to_buffer(|value: &str, output: &mut String| {
        if let Some(first_char) = value.chars().next() {
            write!(output, "{}{}ay", &value[1..], first_char).unwrap()
        }
    });
    Ok(out.into_series())
}

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

#[polars_expr(output_type_func=same_output_type)]
fn noop(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    Ok(s.clone())
}

// TODO estimate median should also work on t-digest and be a shortcut for estimate_quantile with quantile=0.5
#[polars_expr(output_type=Int64)]
fn estimate_median(inputs: &[Series]) -> PolarsResult<Series> {
    let values = &inputs[0].i64()?;
    let t = TDigest::new_with_size(100);
    let chunks: Vec<TDigest> = values
        .downcast_iter()
        .map(|chunk| {
            let array = chunk.as_any().downcast_ref::<Int64Array>().unwrap();
            let val_vec = array
                .values()
                .iter()
                .filter_map(|v| Ok(Some(*v as f64)).transpose())
                .collect::<Result<Vec<f64>, Vec<f64>>>();
            t.merge_unsorted(val_vec.unwrap().to_owned())
        })
        .collect();

    let t_global = TDigest::merge_digests(chunks);
    let ans = t_global.estimate_quantile(0.5);

    Ok(Series::new("", vec![ans]))
}

fn tdigest_output(_: &[Field]) -> PolarsResult<Field> {
    let fields = vec![
        Field::new(
            "centroids",
            DataType::List(Box::new(DataType::Struct(vec![
                Field::new("mean", DataType::Int64),
                Field::new("weight", DataType::Int64),
            ]))),
        ),
        Field::new("sum", DataType::Int64),
        Field::new("min", DataType::Int64),
        Field::new("max", DataType::Int64),
        Field::new("count", DataType::Int64),
        Field::new("max_size", DataType::Int64),
    ];

    Ok(Field::new("tdigest", DataType::Struct(fields)))
}

// fn tidgest_compute<T: NumericNative, PDT: PolarsDataType>(values: &ChunkedArray<PDT>) -> Vec<TDigest> {
//     let chunks: Vec<TDigest> = POOL.install(|| {
//         values
//             .downcast_iter()
//             .par_bridge()
//             .map(|chunk| {
//                 let t = TDigest::new_with_size(100);
//                 let array = chunk.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
//                 let val_vec: Vec<f64> = array
//                     .values()
//                     .iter()
//                     .filter_map(|v| Some(*v. as f64))
//                     .collect();
//                 t.merge_unsorted(val_vec.to_owned())
//             })
//             .collect::<Vec<TDigest>>()
//     });
//     chunks
// }

// Todo support other numerical types
#[polars_expr(output_type_func=tdigest_output)]
fn tdigest(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let chunks = match series.dtype() {
        DataType::Float64 => {
            let values = series.f64()?;
            let chunks: Vec<TDigest> = POOL.install(|| {
                values
                    .downcast_iter()
                    .par_bridge()
                    .map(|chunk| {
                        let t = TDigest::new_with_size(100);
                        let array = chunk.as_any().downcast_ref::<Float64Array>().unwrap();
                        let val_vec: Vec<f64> = array.values().iter().copied().collect();
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
                        let t = TDigest::new_with_size(100);
                        let array = chunk.as_any().downcast_ref::<Float32Array>().unwrap();
                        let val_vec: Vec<f64> =
                            array.values().iter().map(|v| (*v as f64)).collect();
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
                        let t = TDigest::new_with_size(100);
                        let array = chunk.as_any().downcast_ref::<Int64Array>().unwrap();
                        let val_vec: Vec<f64> =
                            array.values().iter().map(|v| (*v as f64)).collect();
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
                        let t = TDigest::new_with_size(100);
                        let array = chunk.as_any().downcast_ref::<Int32Array>().unwrap();
                        let val_vec: Vec<f64> =
                            array.values().iter().map(|v| (*v as f64)).collect();
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

    let td_json = serde_json::to_string(&td_global).unwrap();

    let file = Cursor::new(&td_json);
    let df = JsonReader::new(file)
        .with_json_format(JsonFormat::JsonLines)
        .infer_schema_len(Some(3))
        .with_batch_size(NonZeroUsize::new(3).unwrap())
        .finish()
        .unwrap();

    Ok(df.into_struct(series.name()).into_series())
}

#[polars_expr(output_type_func=tdigest_output)]
fn tdigest_cast(inputs: &[Series]) -> PolarsResult<Series> {
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
                let t = TDigest::new_with_size(100);
                let array = chunk.as_any().downcast_ref::<Float64Array>().unwrap();
                t.merge_unsorted(array.values().to_vec())
            })
            .collect::<Vec<TDigest>>()
    });

    let t_global = TDigest::merge_digests(chunks);

    let td_json = serde_json::to_string(&t_global).unwrap();

    let file = Cursor::new(&td_json);
    let df = JsonReader::new(file)
        .with_json_format(JsonFormat::JsonLines)
        .infer_schema_len(Some(3))
        .with_batch_size(NonZeroUsize::new(3).unwrap())
        .finish()
        .unwrap();

    Ok(df.into_struct(values.name()).into_series())
}

#[derive(Debug, Deserialize)]
struct TDigestCol {
    tdigest: TDigest,
}

#[derive(Debug, Deserialize)]
struct MergeTDKwargs {
    quantile: f64,
}

// TODO this should check the type of the series and also work on series of Type f64
#[polars_expr(output_type=Float64)]
fn estimate_quantile(inputs: &[Series], kwargs: MergeTDKwargs) -> PolarsResult<Series> {
    let mut df = inputs[0].clone().into_frame();
    df.set_column_names(vec!["tdigest"].as_slice()).unwrap();
    let mut buf = BufWriter::new(Vec::new());
    let _json = JsonWriter::new(&mut buf)
        .with_json_format(JsonFormat::Json)
        .finish(&mut df);

    let bytes = buf.into_inner().unwrap();
    let json_str = String::from_utf8(bytes).unwrap();
    let tdigest_json: Vec<TDigestCol> =
        serde_json::from_str(&json_str).expect("Failed to parse the tigest JSON string");

    let tdigests: Vec<TDigest> = tdigest_json.into_iter().map(|td| td.tdigest).collect();
    let tdigest = TDigest::merge_digests(tdigests);
    let ans = tdigest.estimate_quantile(kwargs.quantile);
    Ok(Series::new("", vec![ans]))
}
