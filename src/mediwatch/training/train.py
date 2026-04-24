import argparse
import json

from mediwatch.dataset import load_raw_dataset, make_dataset_bundle
from mediwatch.training.pipeline import train_and_log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--class-weight", type=str, default="balanced")
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    # `bundle` includes all the separate datasets (features, target, train, test) in one class for readability
    bundle = make_dataset_bundle(load_raw_dataset())
    # This CLI intentionally trains the simple logistic-regression incumbent.
    # The retraining DAG can later compare it against richer tuned model families.
    result = train_and_log(
        bundle,
        c_value=args.c_value,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        register_as_best=not args.no_promote,
    )
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
