use crate::tdigest::{Centroid, TDigest};
use polars::prelude::*;
use polars::series::Series;

// TODO: error handling w/o panic
pub fn parse_tdigests(input: &Series) -> Vec<TDigest> {
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

pub fn tdigest_to_series(tdigest: TDigest, name: &str) -> Series {
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
