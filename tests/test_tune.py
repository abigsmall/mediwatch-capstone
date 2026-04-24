from types import SimpleNamespace

from mediwatch.training import tune


def test_trainable_logs_trial_as_child_run(monkeypatch):
    captured = {}

    def fake_train_and_log(
        bundle,
        model_config,
        register_as_best,
        parent_run_id,
        extra_tags,
        run_name_prefix,
    ):
        captured.update(
            {
                "bundle": bundle,
                "model_config": model_config,
                "register_as_best": register_as_best,
                "parent_run_id": parent_run_id,
                "extra_tags": extra_tags,
                "run_name_prefix": run_name_prefix,
            }
        )
        return SimpleNamespace(
            average_precision=0.81,
            f1=0.71,
            precision=0.72,
            recall=0.73,
            roc_auc=0.82,
            positive_rate=0.30,
        )

    monkeypatch.setattr(tune, "train_and_log", fake_train_and_log)

    metrics = tune.trainable(
        {"model_config": {"model_type": "extra_trees", "n_estimators": 10}},
        bundle=object(),
        parent_run_id="parent-123",
    )

    assert captured["model_config"] == {"model_type": "extra_trees", "n_estimators": 10}
    assert captured["register_as_best"] is False
    assert captured["parent_run_id"] == "parent-123"
    assert captured["extra_tags"] == {
        "run_type": "tuning_trial",
        "model_type": "extra_trees",
    }
    assert captured["run_name_prefix"] == "mediwatch-tune-trial-extra_trees"
    assert metrics["average_precision"] == 0.81
