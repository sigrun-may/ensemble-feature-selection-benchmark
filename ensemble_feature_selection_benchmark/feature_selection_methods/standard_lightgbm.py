# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""LightGBM training."""
import os

import logging
import warnings

import lightgbm as lgb
import numpy as np
import shap


# _logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.basicConfig(format="%(levelname)s:%(message)s")
logger = logging.getLogger(__name__)
# logger.setLevel(settings.logging.level)
logger.setLevel("ERROR")


def calculate_score(data_inner_cv_iteration, parameters):
    train_df, validation_df, _ = data_inner_cv_iteration

    # prepare train data
    y_train = train_df["label"].values - 1
    x_train = train_df.loc[:, train_df.columns != "label"]

    # prepare validation data
    y_validation = validation_df["label"].values - 1
    x_validation = validation_df.loc[:, validation_df.columns != "label"]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = [lgb.Dataset(x_validation, y_validation)]

    # set cpu count for parallelization
    parameters["num_threads"] = os.cpu_count()

    # build model
    eval_result = {}
    model = lgb.train(
        parameters,
        lgb_train,
        valid_sets=lgb_eval,
        callbacks=[lgb.record_evaluation(eval_result=eval_result)],
        verbose_eval=False,
    )

    # predicted_y_validation = model.predict(x_validation)
    score = min(eval_result["valid_0"]["binary_logloss"])

    # calculate shap values
    explainer = shap.explainers.Tree(model)
    # TODO train oder validation
    # prepare validation data
    x_validation = validation_df.loc[:, validation_df.columns != "label"]
    shap_values = explainer(x_validation)
    # https://github.com/slundberg/shap/issues/933
    added_shap_values = np.sum(np.abs(shap_values.values), axis=0)[:, 0]
    shap_list = added_shap_values.tolist()
    return score, list(model.feature_importance(importance_type="gain")), shap_list


def calculate_micro_feature_importance(train_data_outer_cv_df, hyperparameters_dict):
    # prepare train data
    y_train = train_data_outer_cv_df["label"].values
    x_train = train_data_outer_cv_df.loc[:, train_data_outer_cv_df.columns != "label"]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(x_train, y_train)

    # build model
    model = lgb.train(
        hyperparameters_dict,
        lgb_train,
        verbose_eval=False,
    )
    return list(model.feature_importance(importance_type="gain"))
