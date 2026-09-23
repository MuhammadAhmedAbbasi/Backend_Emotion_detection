"""Train and compare EEG emotion classifiers."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# Paths
ALGORITHM_DIR = Path(__file__).resolve().parent
FEATURES_FILE = ALGORITHM_DIR / "features" / "features.npz"
TRAINED_MODEL_FILE = ALGORITHM_DIR / "saved_model" / "trained_model.pkl"


# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_SELECTION_METRIC = "macro_f1"
TREE_ESTIMATORS = 300

SELECTED_SEED_CHANNELS = ["TP7", "TP8", "AF3", "AF4"]
SELECTED_MUSE_CHANNELS = ["TP9", "TP10", "AF7", "AF8"]

LABEL_NAMES = {
    0: "neutral",
    1: "sad",
    2: "fear",
    3: "happy",
}


def load_features(
    features_file: Path = FEATURES_FILE,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, Any],
]:
    """Load extracted features and their metadata."""
    if not features_file.exists():
        raise FileNotFoundError(
            f"Feature file not found at {features_file}. "
            "Run feature_extraction.py first."
        )

    data = np.load(features_file, allow_pickle=True)

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)

    groups = (
        np.asarray(data["groups"], dtype=np.int64)
        if "groups" in data.files
        else np.arange(len(y))
    )

    feature_names = [
        str(name)
        for name in data["feature_names"]
    ]

    metadata = (
        json.loads(str(data["metadata_json"]))
        if "metadata_json" in data.files
        else {}
    )

    return X, y, groups, feature_names, metadata


def build_classifiers(
    random_state: int = RANDOM_STATE,
    include_svm: bool = True,
    include_optional_boosters: bool = True,
    tree_estimators: int = TREE_ESTIMATORS,
) -> dict[str, Any]:
    """Create the classifiers that will be compared."""
    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=tree_estimators,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=tree_estimators,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if include_svm:
        classifiers["linear_svm"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(
                            C=0.5,
                            dual=False,
                            max_iter=20000,
                            tol=1e-3,
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                        cv=3,
                    ),
                ),
            ]
        )

    if include_optional_boosters:
        add_optional_boosters(
            classifiers,
            random_state,
            tree_estimators,
        )

    return classifiers


def add_optional_boosters(
    classifiers: dict[str, Any],
    random_state: int,
    tree_estimators: int,
) -> None:
    """Add XGBoost and LightGBM when those packages are installed."""
    try:
        from xgboost import XGBClassifier

        classifiers["xgboost"] = XGBClassifier(
            n_estimators=tree_estimators,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception as error:
        print(f"Skipping XGBoost: {error}")

    try:
        from lightgbm import LGBMClassifier

        classifiers["lightgbm"] = LGBMClassifier(
            n_estimators=tree_estimators,
            learning_rate=0.05,
            num_leaves=31,
            objective="multiclass",
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except Exception as error:
        print(f"Skipping LightGBM: {error}")


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Calculate the main evaluation metrics for one trained model."""
    predictions = model.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    label_ids = sorted(LABEL_NAMES)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=label_ids,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=label_ids,
            target_names=[
                LABEL_NAMES[label_id]
                for label_id in label_ids
            ],
            zero_division=0,
        ),
    }


def print_metrics(name: str, metrics: dict[str, Any]) -> None:
    """Print model evaluation results in a readable format."""
    print(f"\n{name}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['macro_precision']:.4f}")
    print(f"Recall:    {metrics['macro_recall']:.4f}")
    print(f"F1-score:  {metrics['macro_f1']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print("Classification Report:")
    print(metrics["classification_report"])


def train_and_select_model(
    features_file: Path = FEATURES_FILE,
    model_file: Path = TRAINED_MODEL_FILE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    include_svm: bool = True,
    include_optional_boosters: bool = True,
    tree_estimators: int = TREE_ESTIMATORS,
) -> Path:
    """Train all enabled classifiers and save the best one."""
    X, y, groups, feature_names, metadata = load_features(features_file)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "Training requires at least two emotion classes."
        )

    X_train, X_test, y_train, y_test, _, _ = train_test_split(
        X,
        y,
        groups,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    classifiers = build_classifiers(
        random_state=random_state,
        include_svm=include_svm,
        include_optional_boosters=include_optional_boosters,
        tree_estimators=tree_estimators,
    )

    results = {}
    trained_models = {}

    print(
        f"Training samples: {len(X_train)}, "
        f"test samples: {len(X_test)}"
    )
    print(f"Feature count: {X_train.shape[1]}")

    for name, model in classifiers.items():
        print(f"\nTraining {name}...")

        try:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
        except Exception as error:
            print(f"Skipping {name}: {error}")
            continue

        trained_models[name] = model
        results[name] = metrics
        print_metrics(name, metrics)

    if not results:
        raise RuntimeError(
            "No classifier trained successfully."
        )

    best_name = max(
        results,
        key=lambda name: (
            results[name][MODEL_SELECTION_METRIC],
            results[name]["accuracy"],
        ),
    )

    best_model = trained_models[best_name]
    best_score = results[best_name][MODEL_SELECTION_METRIC]

    print(
        f"\nBest model: {best_name} "
        f"({MODEL_SELECTION_METRIC}={best_score:.4f})"
    )

    model_file.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_names,
        "label_names": LABEL_NAMES,
        "selected_seed_channels": metadata.get(
            "selected_seed_channels",
            SELECTED_SEED_CHANNELS,
        ),
        "selected_muse_channels": metadata.get(
            "selected_muse_channels",
            SELECTED_MUSE_CHANNELS,
        ),
        "sampling_rate_hz": metadata.get(
            "sampling_rate_hz",
            200,
        ),
        "metrics": results,
        "source_features_file": str(features_file),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(artifact, model_file)

    print(f"Saved best model to: {model_file}")
    return model_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare EEG emotion classifiers."
    )

    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURES_FILE,
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=TRAINED_MODEL_FILE,
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=TEST_SIZE,
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
    )
    parser.add_argument(
        "--skip-svm",
        action="store_true",
        help="Skip the calibrated linear SVM candidate.",
    )
    parser.add_argument(
        "--skip-optional-boosters",
        action="store_true",
        help="Skip XGBoost and LightGBM.",
    )
    parser.add_argument(
        "--tree-estimators",
        type=int,
        default=TREE_ESTIMATORS,
        help="Number of trees or boosting rounds.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_and_select_model(
        features_file=args.features,
        model_file=args.model_output,
        test_size=args.test_size,
        random_state=args.random_state,
        include_svm=not args.skip_svm,
        include_optional_boosters=not args.skip_optional_boosters,
        tree_estimators=args.tree_estimators,
    )


if __name__ == "__main__":
    main()
