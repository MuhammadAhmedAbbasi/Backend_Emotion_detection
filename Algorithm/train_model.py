"""Train lightweight emotion classifiers on extracted EEG features."""

from __future__ import annotations

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

from Algorithm import config


def load_features(features_file: Path = config.FEATURES_FILE) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if not features_file.exists():
        raise FileNotFoundError(f"Feature file not found at {features_file}. Run feature_extraction.py first.")

    data = np.load(features_file, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    groups = np.asarray(data["groups"], dtype=np.int64) if "groups" in data.files else np.arange(len(y))
    feature_names = [str(name) for name in data["feature_names"]]
    metadata = json.loads(str(data["metadata_json"])) if "metadata_json" in data.files else {}
    return X, y, groups, feature_names, metadata


def build_classifiers(
    random_state: int = config.RANDOM_STATE,
    include_svm: bool = True,
    include_optional_boosters: bool = True,
    tree_estimators: int = 300,
) -> dict[str, Any]:
    """Create candidate classifiers. XGBoost/LightGBM are included only if installed."""
    classifiers: dict[str, Any] = {
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
            steps=[
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
        except Exception as exc:
            print(f"Skipping XGBoost: {exc}")

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
        except Exception as exc:
            print(f"Skipping LightGBM: {exc}")

    return classifiers


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    y_pred = model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
            labels=sorted(config.LABEL_NAMES),
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            labels=sorted(config.LABEL_NAMES),
            target_names=[config.LABEL_NAMES[label] for label in sorted(config.LABEL_NAMES)],
            zero_division=0,
        ),
    }


def train_and_select_model(
    features_file: Path = config.FEATURES_FILE,
    model_file: Path = config.TRAINED_MODEL_FILE,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
    include_svm: bool = True,
    include_optional_boosters: bool = True,
    tree_estimators: int = 300,
) -> Path:
    X, y, groups, feature_names, metadata = load_features(features_file)
    if len(np.unique(y)) < 2:
        raise ValueError("Training requires at least two emotion classes.")

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
    results: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}

    print(f"Training samples: {X_train.shape[0]}, test samples: {X_test.shape[0]}")
    print(f"Feature count: {X_train.shape[1]}")

    for name, model in classifiers.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            continue
        results[name] = metrics
        trained_models[name] = model
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['macro_precision']:.4f}")
        print(f"Recall:    {metrics['macro_recall']:.4f}")
        print(f"F1-score:  {metrics['macro_f1']:.4f}")
        print("Confusion Matrix:")
        print(np.array(metrics["confusion_matrix"]))
        print("Classification Report:")
        print(metrics["classification_report"])

    if not results:
        raise RuntimeError("No classifier trained successfully.")

    best_name = max(
        results,
        key=lambda model_name: (
            results[model_name][config.MODEL_SELECTION_METRIC],
            results[model_name]["accuracy"],
        ),
    )
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name} ({config.MODEL_SELECTION_METRIC}={results[best_name][config.MODEL_SELECTION_METRIC]:.4f})")

    model_file.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": feature_names,
        "label_names": config.LABEL_NAMES,
        "selected_seed_channels": config.SELECTED_SEED_CHANNELS,
        "selected_muse_channels": config.SELECTED_MUSE_CHANNELS,
        "sampling_rate_hz": metadata.get("sampling_rate_hz", config.SAMPLING_RATE),
        "metrics": results,
        "source_features_file": str(features_file),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, model_file)
    print(f"Saved best model to: {model_file}")
    return model_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare EEG emotion classifiers.")
    parser.add_argument("--features", type=Path, default=config.FEATURES_FILE)
    parser.add_argument("--model-output", type=Path, default=config.TRAINED_MODEL_FILE)
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=config.RANDOM_STATE)
    parser.add_argument("--skip-svm", action="store_true", help="Skip the calibrated linear SVM candidate.")
    parser.add_argument(
        "--skip-optional-boosters",
        action="store_true",
        help="Skip XGBoost and LightGBM even if they are installed.",
    )
    parser.add_argument("--tree-estimators", type=int, default=300, help="Number of trees/boosting rounds.")
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
