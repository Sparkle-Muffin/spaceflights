"""
This is a boilerplate pipeline 'dense_model'
generated using Kedro 0.19.9
"""

import logging
import typing as t

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras import models, layers # type: ignore

def split_data(df: pd.DataFrame, parameters: t.Dict) -> t.Tuple:
    X = df[parameters["features"]]
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series):
    model = models.Sequential([layers.Dense(10), layers.Dense(1)])
    model.compile(loss="mae", optimizer="adam", metrics=["mae"])
    model.fit(x=X_train, y=y_train, epochs=100)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = model.predict(X_test)
    score = r2_score(y_true=y_test, y_pred=y_pred)
    logger = logging.getLogger(__name__)
    logger.info("Model has a coefficient R^2 of %.3f on test data.", score)
