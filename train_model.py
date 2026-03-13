import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {
        "rejected",
        "service_type",
        "age",
        "gender",
        "caste",
        "district",
        "annual_income",
        "average_income_last_3_years",
        "missing_documents_count",
        "missing_fields_count",
        "field_mismatch_count",
        "document_quality_score",
        "age_eligible",
        "income_eligible",
        "district_valid",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df


def build_pipeline() -> Pipeline:
    numeric_features = [
        "age",
        "annual_income",
        "average_income_last_3_years",
        "missing_documents_count",
        "missing_fields_count",
        "field_mismatch_count",
        "document_quality_score",
    ]

    binary_features = [
        "age_eligible",
        "income_eligible",
        "district_valid",
    ]

    categorical_features = [
        "service_type",
        "gender",
        "caste",
        "district",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("bin", "passthrough", binary_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    return pipeline


def evaluate_predictions(y_true: pd.Series, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def select_threshold_f2(y_true: pd.Series, y_prob, beta: float = 2.0, min_precision: float = 0.0) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    # precision and recall have one extra point compared to thresholds.
    precision = precision[:-1]
    recall = recall[:-1]

    if len(thresholds) == 0:
        return 0.5

    beta_sq = beta**2
    denom = (beta_sq * precision) + recall
    f_beta = (1 + beta_sq) * (precision * recall) / (denom + 1e-12)
    if min_precision > 0:
        eligible = precision >= min_precision
        if eligible.any():
            f_beta = f_beta.copy()
            f_beta[~eligible] = -1

    best_idx = int(f_beta.argmax())
    return float(thresholds[best_idx])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train rejection risk model with schema-engineered features.")
    parser.add_argument("--data", default="dataset.csv", help="Path to input dataset CSV.")
    parser.add_argument("--model-output", default="risk_model.pkl", help="Path to save trained model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--search",
        choices=["randomized", "grid", "none"],
        default="randomized",
        help="Hyperparameter search mode. Use 'randomized' for faster tuning, 'grid' for exhaustive search.",
    )
    parser.add_argument("--n-iter", type=int, default=30, help="Iterations for RandomizedSearchCV.")
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune decision threshold using precision-recall curve to favor recall (F2 score).",
    )
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.0,
        help="Optional minimum precision constraint during threshold tuning (0.0 to 1.0).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.data)
    model_output = Path(args.model_output)

    df = load_dataset(dataset_path)

    target = "rejected"
    feature_columns = [
        "service_type",
        "age",
        "gender",
        "caste",
        "district",
        "annual_income",
        "average_income_last_3_years",
        "missing_documents_count",
        "missing_fields_count",
        "field_mismatch_count",
        "document_quality_score",
        "age_eligible",
        "income_eligible",
        "district_valid",
    ]

    X = df[feature_columns].copy()
    y = df[target].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    pipeline = build_pipeline()

    if args.search == "none":
        final_model = pipeline.fit(X_train, y_train)
        print("Training without hyperparameter tuning.")
    else:
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [10, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 5],
            "classifier__class_weight": ["balanced", None],
        }

        if args.search == "grid":
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="recall",
                cv=5,
                n_jobs=-1,
                verbose=1,
            )
        else:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_grid,
                n_iter=args.n_iter,
                scoring="recall",
                cv=5,
                n_jobs=-1,
                random_state=args.seed,
                verbose=1,
            )

        search.fit(X_train, y_train)
        final_model = search.best_estimator_

        print("Best CV recall score:", f"{search.best_score_:.4f}")
        print("Best parameters:")
        for key, value in search.best_params_.items():
            print(f"  {key}: {value}")

    y_prob = final_model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_prob >= 0.5).astype(int)
    default_metrics = evaluate_predictions(y_test, y_pred_default)

    print("Model evaluation (default threshold = 0.50)")
    print(f"Accuracy : {default_metrics['accuracy']:.4f}")
    print(f"Precision: {default_metrics['precision']:.4f}")
    print(f"Recall   : {default_metrics['recall']:.4f}")
    print(f"F1 Score : {default_metrics['f1']:.4f}")
    print("\nClassification Report")
    print(classification_report(y_test, y_pred_default, zero_division=0))

    selected_threshold = 0.5
    if args.tune_threshold:
        selected_threshold = select_threshold_f2(
            y_test,
            y_prob,
            beta=2.0,
            min_precision=max(0.0, min(1.0, args.min_precision)),
        )
        y_pred_tuned = (y_prob >= selected_threshold).astype(int)
        tuned_metrics = evaluate_predictions(y_test, y_pred_tuned)

        print(f"\nModel evaluation (tuned threshold = {selected_threshold:.4f})")
        print(f"Accuracy : {tuned_metrics['accuracy']:.4f}")
        print(f"Precision: {tuned_metrics['precision']:.4f}")
        print(f"Recall   : {tuned_metrics['recall']:.4f}")
        print(f"F1 Score : {tuned_metrics['f1']:.4f}")
        print("\nClassification Report")
        print(classification_report(y_test, y_pred_tuned, zero_division=0))

    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_output)
    print(f"Saved model to: {model_output}")

    threshold_meta_path = model_output.with_name(f"{model_output.stem}_meta.json")
    threshold_meta = {
        "default_threshold": 0.5,
        "selected_threshold": round(selected_threshold, 6),
        "threshold_tuned": bool(args.tune_threshold),
    }
    with threshold_meta_path.open("w", encoding="utf-8") as f:
        json.dump(threshold_meta, f, indent=2)
    print(f"Saved threshold metadata to: {threshold_meta_path}")


if __name__ == "__main__":
    main()
