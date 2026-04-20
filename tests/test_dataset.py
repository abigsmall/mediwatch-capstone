from mediwatch.dataset import build_feature_frame, normalize_raw_dataframe


def test_target_mapping_for_readmission():
    df = normalize_raw_dataframe(
        __import__("pandas").DataFrame(
            [
                {"readmitted": "NO", "time_in_hospital": 3},
                {"readmitted": "<30", "time_in_hospital": 5},
            ]
        )
    )
    _, target = build_feature_frame(df)
    assert target.tolist() == [0, 1]
