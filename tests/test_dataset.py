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


def test_target_columns_are_not_in_feature_frame():
    df = normalize_raw_dataframe(
        __import__("pandas").DataFrame(
            [
                {
                    "encounter_id": "1",
                    "patient_nbr": "2",
                    "readmitted": "NO",
                    "time_in_hospital": 3,
                }
            ]
        )
    )
    features, target = build_feature_frame(df)
    assert "readmitted" not in features.columns
    assert "readmitted_binary" not in features.columns
    assert target.tolist() == [0]
