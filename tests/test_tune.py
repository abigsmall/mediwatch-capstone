from types import SimpleNamespace

from mediwatch.training import tune


def test_search_space_is_trimmed_and_includes_catboost():
    assert len(tune.SEARCH_SPACE) < 40
    assert {config["model_type"] for config in tune.SEARCH_SPACE} == {
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "catboost",
    }
    assert all(
        config.get("max_depth") is not None
        for config in tune.SEARCH_SPACE
        if config["model_type"] in {"random_forest", "extra_trees"}
    )
    assert all(
        config.get("min_samples_leaf", 0) >= 3
        for config in tune.SEARCH_SPACE
        if config["model_type"] in {"random_forest", "extra_trees"}
    )


def test_init_ray_for_tuning_uses_dashboard_settings(monkeypatch):
    captured = {}

    monkeypatch.setattr(tune.ray, "is_initialized", lambda: False)

    def fake_ray_init(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(tune.ray, "init", fake_ray_init)

    assert tune._init_ray_for_tuning() is True
    assert captured == {
        "include_dashboard": True,
        "dashboard_host": tune.settings.ray_dashboard_host,
        "dashboard_port": tune.settings.ray_dashboard_port,
        "object_store_memory": tune.settings.ray_object_store_memory_bytes,
    }


def test_init_ray_for_tuning_reuses_existing_ray(monkeypatch):
    monkeypatch.setattr(tune.ray, "is_initialized", lambda: True)

    def fail_if_called(**kwargs):
        raise AssertionError("ray.init should not be called")

    monkeypatch.setattr(tune.ray, "init", fail_if_called)

    assert tune._init_ray_for_tuning() is False


def test_trainable_logs_trial_as_child_run(monkeypatch):
    captured = {}

    def fake_train_and_log(
        bundle,
        model_config,
        register_as_best,
        parent_run_id,
        extra_tags,
        run_name_prefix,
        log_model_artifacts,
    ):
        captured.update(
            {
                "bundle": bundle,
                "model_config": model_config,
                "register_as_best": register_as_best,
                "parent_run_id": parent_run_id,
                "extra_tags": extra_tags,
                "run_name_prefix": run_name_prefix,
                "log_model_artifacts": log_model_artifacts,
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
    assert captured["log_model_artifacts"] is False
    assert metrics["average_precision"] == 0.81
