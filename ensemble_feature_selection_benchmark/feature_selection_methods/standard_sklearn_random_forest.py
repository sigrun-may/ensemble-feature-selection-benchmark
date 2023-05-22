# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Sklearn Random Forest Classifier training."""

import logging

import numpy as np
import shap
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier


_logger = logging.getLogger(__name__)


def train_random_forest(train_df, params):
    # prepare train data
    y_train = train_df["label"].values
    x_train = train_df.loc[:, train_df.columns != "label"]

    with parallel_backend(backend="loky", n_jobs=1, inner_max_num_threads=1):
        # build model
        model = RandomForestClassifier()
        # if fixed_parameters:
        #     params = fixed_parameters
        # elif trial:
        #     assert isinstance(trial, optuna.trial._trial.Trial)
        #     # TODO set seed
        #     params = {"n_estimators": trial.suggest_int("n_estimators", 1, 150), "random_state": 42}
        # else:
        #     raise ValueError("No parameters available")
        model = model.set_params(**params)
        model.fit(x_train, np.ravel(y_train))
    return model


def calculate_score(data_inner_cv_iteration, parameters):
    train_df, validation_df, _ = data_inner_cv_iteration

    # prepare validation data
    x_validation = validation_df.loc[:, validation_df.columns != "label"]
    y_validation = np.ravel(validation_df["label"])

    # build model
    model = train_random_forest(train_df, parameters)
    # predicted_y_validation = model.predict(x_validation)
    score = model.score(x_validation, y_validation)

    # calculate shap values
    explainer = shap.explainers.Tree(
        model, train_df.loc[:, train_df.columns != "label"]
    )
    # TODO train oder validation
    shap_values = explainer(x_validation)
    # https://github.com/slundberg/shap/issues/933
    added_shap_values = np.sum(np.abs(shap_values.values), axis=0)[:, 0]
    shap_list = added_shap_values.tolist()
    return score, list(model.feature_importances_), shap_list


def calculate_micro_feature_importance(train_data_outer_cv_df, hyperparameters_dict):
    assert len(hyperparameters_dict) > 0
    # build model for micro_feature_importance
    micro_model = train_random_forest(train_data_outer_cv_df, hyperparameters_dict)
    return list(micro_model.feature_importances_)
