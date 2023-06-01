# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Sklearn Lasso training."""


import logging

import numpy as np
import shap
from joblib import parallel_backend
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score


_logger = logging.getLogger(__name__)


def train_lasso_sklearn(x_train, y_train, parameters):
    """Trains the Lasso model using sklearn.

    Args:
        x_train: DataFrame containing the data to train the model excluding the targets.
        y_train: Array or DataFrame containing the targets for the model fitting.
        parameters: Dictionary containing the value for the parameter the model should be trained with.

    Returns:
        A trained Lasso model.

    """
    with parallel_backend(backend="loky", n_jobs=1, inner_max_num_threads=1):
        # build LASSO model
        model = Lasso(
            alpha=parameters["alpha"],
            fit_intercept=True,
            positive=False,
        )
        model.fit(np.asfortranarray(x_train), y_train)
    return model


def calculate_score(data_inner_cv_iteration, parameters):
    """Trains a Lasso Regression Model and calculates scores of this model.

    After training the model this function calculates the r2-score, model coefficients (weight vectors) and
    shap-values of this model.

    Args:
        data_inner_cv_iteration: DataSplit containing the DataFrame the model to be scored should be trained on.
        parameters: Dictionary containing the parameters the model to be scored should be trained with.

    Returns:
        R2-score of the model.
        A list containing the model coefficients (weight vectors, here macro feature importances).
        A list containing the models shap values.

    """
    train_df, validation_df, _ = data_inner_cv_iteration

    # prepare train data
    y_train = train_df["label"].values.reshape(-1, 1)
    x_train = train_df.loc[:, train_df.columns != "label"]

    # prepare validation data
    x_validation = validation_df.loc[:, validation_df.columns != "label"].values
    y_validation = validation_df["label"].values

    # build LASSO model
    lasso_model = train_lasso_sklearn(x_train, y_train, parameters)

    predicted_y_validation = lasso_model.predict(x_validation)
    score = r2_score(y_validation, predicted_y_validation)

    # calculate shap values
    explainer = shap.explainers.Linear(
        lasso_model, train_df.loc[:, train_df.columns != "label"]
    )
    # TODO train oder validation
    shap_values = explainer(x_validation)
    # https://github.com/slundberg/shap/issues/933
    added_shap_values = np.sum(np.abs(shap_values.values), axis=0)
    shap_list = added_shap_values.tolist()
    return score, list(lasso_model.coef_), shap_list


def calculate_micro_feature_importance(train_data_outer_cv_df, hyperparameters_dict):
    """Calculates the micro feature importances using Lasso Regression.

    Args:
        train_data_outer_cv_df:  DataFrame to train the Lasso model on to calculate the micro feature importance.
        hyperparameters_dict: Dictionary containing the hyperparameter "alpha".

    Returns:
        List containing the micro feature importance per feature.

    Raises:
        AssertionError: If no hyperparameters are given.

    """
    # prepare train data
    y_train = train_data_outer_cv_df["label"].values.reshape(-1, 1)
    x_train = train_data_outer_cv_df.loc[:, train_data_outer_cv_df.columns != "label"]
    assert len(hyperparameters_dict) > 0
    assert "alpha" in hyperparameters_dict.keys()
    # build LASSO model for micro_feature_importance
    lasso_micro_model = train_lasso_sklearn(x_train, y_train, hyperparameters_dict)
    return list(lasso_micro_model.coef_)
