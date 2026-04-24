import pandas as pd
import pytest

from mediwatch.training.pipeline import build_pipeline


def _training_frame() -> tuple[pd.DataFrame, pd.Series]:
    return (
        pd.DataFrame(
            [
                {"age": "[50-60)", "time_in_hospital": 2, "num_medications": 7},
                {"age": "[60-70)", "time_in_hospital": 6, "num_medications": 12},
                {"age": "[70-80)", "time_in_hospital": 4, "num_medications": 9},
                {"age": "[80-90)", "time_in_hospital": 8, "num_medications": 15},
            ]
        ),
        pd.Series([0, 1, 0, 1]),
    )


@pytest.mark.parametrize(
    "model_config",
    [
        {
            "model_type": "random_forest",
            "n_estimators": 10,
            "max_depth": 4,
            "min_samples_leaf": 1,
            "class_weight": None,
        },
        {
            "model_type": "extra_trees",
            "n_estimators": 10,
            "max_depth": 4,
            "min_samples_leaf": 1,
            "class_weight": None,
        },
        {
            "model_type": "catboost",
            "iterations": 5,
            "depth": 2,
            "learning_rate": 0.1,
            "l2_leaf_reg": 3.0,
            "auto_class_weights": "Balanced",
            "thread_count": 1,
        },
    ],
)
def test_non_linear_pipeline_configs_fit_and_predict_probabilities(model_config):
    train_x, train_y = _training_frame()
    model = build_pipeline(train_x, model_config=model_config)

    model.fit(train_x, train_y)
    probabilities = model.predict_proba(train_x)[:, 1]

    assert probabilities.shape == (len(train_x),)
    assert all(0 <= probability <= 1 for probability in probabilities)


def test_unknown_model_type_raises_clear_error():
    train_x, _ = _training_frame()

    with pytest.raises(ValueError, match="Unsupported model_type"):
        build_pipeline(train_x, model_config={"model_type": "svm"})
