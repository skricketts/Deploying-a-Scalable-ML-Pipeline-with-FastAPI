import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship", 
    "race", 
    "sex",
    "native-country"
]
LABEL = "salary"


def _sample_test_df(n=500):
    """
    Creates a small, random subset of data for unit tests.
    """
    df = pd.read_csv("data/census.csv")
    return df.sample(n = min(n, len(df)), random_state = 42).reset_index(drop = True)


# Test One
def test_data_shapes_types():
    """
    Data processing returns the expected shapes/types and binary labels.
    """
    df = _sample_test_df()
    train_df, _ = train_test_split(df, test_size = 0.2, random_state = 42)

    X, y, encoder, lb = process_data(
        train_df, categorical_features = CAT_FEATURES, label = LABEL, training = True
    )

    assert X.shape[0] == len(y)
    assert set(y).issubset({0, 1})
    assert hasattr(encoder, "categories_")


# Test Two
def test_model_algorithm_predictions():
    """
    Uses the expected algorithm and produces 0/1 predictions. 
    """
    df = _sample_test_df()
    train_df, test_df = train_test_split(df, test_size = 0.2, random_state = 0)

    X_train, y_train, encoder, lb = process_data(
        train_df, categorical_features = CAT_FEATURES, label = LABEL, training = True
    )
    X_test, y_test, _, _ = process_data(
        test_df, categorical_features = CAT_FEATURES, label = LABEL, training = False, encoder = encoder, lb = lb
    )

    model = train_model(X_train, y_train)
    assert isinstance(model, LogisticRegression)

    preds = inference(model, X_test)
    assert preds.shape == y_test.shape
    assert set(preds).issubset({0, 1})

    p, r, f1 = compute_model_metrics(y_test, preds)
    for v in (p, r, f1):
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0


# Test Three
def test_train_test_split_ratio():
    """
    # Check that data split produces roughly the same sizes.
    """
    df = _sample_test_df()
    train, test = train_test_split(df, test_size = 0.2, random_state = 42)

    total = len(df)
    assert abs(len(test) / total - 0.2) < 0.05
    assert abs(len(train) / total - 0.8) < 0.05
