use criterion::{black_box, criterion_group, criterion_main, Criterion};
use polars::series::Series;
use polars_tdigest::tdigest::{
    codecs::parse_tdigests, codecs::tdigest_to_series, Centroid, TDigest,
};

mod legacy_json_parser {
    use std::io::BufWriter;

    use polars::series::Series;
    use polars_tdigest::tdigest::TDigest;
    use serde::Deserialize;

    use polars::prelude::*;

    #[derive(Debug, Deserialize)]
    struct TDigestCol {
        tdigest: TDigest,
    }

    fn extract_tdigest_vec(inputs: &Series) -> Vec<TDigestCol> {
        let mut df = inputs.clone().into_frame();
        df.set_column_names(vec!["tdigest"].as_slice()).unwrap();
        let mut buf = BufWriter::new(Vec::new());
        let _json = JsonWriter::new(&mut buf)
            .with_json_format(JsonFormat::Json)
            .finish(&mut df);

        let bytes = buf.into_inner().unwrap();
        let json_str = String::from_utf8(bytes).unwrap();

        serde_json::from_str(&json_str).expect("Failed to parse the tdigest JSON string")
    }

    pub fn parse(series: &Series) -> Vec<TDigest> {
        let tdigest_json: Vec<TDigestCol> = extract_tdigest_vec(series);
        tdigest_json.into_iter().map(|td| td.tdigest).collect()
    }
}

fn create_tdigest(size: usize, nonce: f64) -> TDigest {
    let centroids: Vec<Centroid> = (1..size)
        .map(|n| Centroid::new((n as f64 + nonce) * 1000.0, n as f64 + nonce))
        .collect();
    TDigest::new(centroids, nonce, size as f64, 42.0, 42.0, size * 10)
}

fn create_series() -> Series {
    let mut series = tdigest_to_series(create_tdigest(100, 0.0), "name");
    (1..9).for_each(|i| {
        series
            .append(&tdigest_to_series(create_tdigest(100, 0.0), "name"))
            .unwrap();
    });
    return series;
}

fn reading_tdigests_directly_benchmark(c: &mut Criterion) {
    let series = create_series();
    c.bench_function("parse", |b| b.iter(|| parse_tdigests(black_box(&series))));
}

fn reading_tdigests_json_benchmark(c: &mut Criterion) {
    let series = create_series();

    c.bench_function("parse_via_serde", |b| {
        b.iter(|| legacy_json_parser::parse(black_box(&series)))
    });
}

fn sleep_bench(c: &mut Criterion) {
    use std::{thread, time};

    let ten_millis = time::Duration::from_millis(10);

    c.bench_function("sleep", |b| b.iter(|| thread::sleep(ten_millis)));
}

criterion_group!(
    benches,
    reading_tdigests_directly_benchmark,
    reading_tdigests_json_benchmark,
    sleep_bench
);
criterion_main!(benches);
