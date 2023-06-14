# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Embedded feature selection with optuna hyperparameter optimization."""
import math

import logging

import numpy as np
import optuna
from optuna.samplers import TPESampler

from config import settings


_logger = logging.getLogger(__name__)


def _set_results(preprocessed_data, outer_cv_loop, selection_method, study) -> dict:
    result_dict = {}

    # check if at least one trial was finished
    if len(study.best_trials) > 0:
        result_dict[
            "micro_feature_importance"
        ] = selection_method.calculate_micro_feature_importance(
            preprocessed_data.outer_preprocessed_data_splits[
                outer_cv_loop
            ].train_data_outer_cv_df,
            study.best_params,
        )
        result_dict["validation_metric_value"] = study.best_value
        result_dict["best_hyper_parameters"] = study.best_params

        assert "shap_values" in study.best_trial.user_attrs.keys()
        assert "macro_feature_importances" in study.best_trial.user_attrs.keys()
        result_dict["shap_values"] = study.best_trial.user_attrs["shap_values"]
        result_dict["macro_feature_importances"] = study.best_trial.user_attrs[
            "macro_feature_importances"
        ]
        assert len(result_dict["macro_feature_importances"]) == len(
            preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_loop]
        )

    else:
        _logger.info(
            "No trial was completed within the hyperparameter optimization! "
            "Please continue HPO and either calculate more trials "
            "or prune less aggressively."
        )
    return result_dict


def select_features(
    preprocessed_data,
    outer_cv_iteration,
    n_trials,
    direction,
    selection_method,
    boosting_type=None,
) -> dict:
    """Select feature subset with given feature selection method.

    Args:
        preprocessed_data: yj +pearson train data
        outer_cv_iteration: index of outer cross-validation loop
        n_trials: number of trials for the hyperparameter optimization for the embedded feature selection
        direction: direction for optimization -> can be "maximize" or "minimize"
        selection_method: method for embedded feature selection
        boosting_type: "gbdt", traditional Gradient Boosting Decision Tree, aliases: "gbrt"
                       "rf", Random Forest, aliases: "random_forest"
                       "extra_trees"

    Returns: selected features + weights
    """

    def optuna_objective(trial):
        # select hyperparemters for the different embedded feature selection methods for ensemble feature selection
        if "lasso" in selection_method.__name__:
            hyperparameter_dict = {
                "alpha": trial.suggest_float("alpha", 0.01, 1.0, log=True),
                "random_state": 42,
            }
        elif "random_forest" in selection_method.__name__:
            hyperparameter_dict = {
                "random_state": 42,
                "criterion": "entropy",
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    2,
                    math.floor(settings.data.number_of_samples / 2),
                ),
                "n_jobs": settings.parallel_processes.n_jobs_training,
            }
        elif "svm" in selection_method.__name__:
            hyperparameter_dict = {
                "C": trial.suggest_float("C", 1, 10),
                "random_state": 42,
            }
        elif "lightgbm" in selection_method.__name__:
            if boosting_type == "extra_trees":
                hyperparameter_dict = {
                    "extra_trees": True,
                    "boosting_type": "random_forest",
                }
            else:
                hyperparameter_dict = {
                    "extra_trees": False,
                    "boosting_type": boosting_type,
                }

            # parameters for model training to combat overfitting
            hyperparameter_dict.update(
                dict(
                    min_data_in_leaf=trial.suggest_int(
                        "min_data_in_leaf",
                        2,
                        math.floor(settings.data.number_of_samples / 2),
                    ),
                    lambda_l1=trial.suggest_float("lambda_l1", 0.0, 3),
                    min_gain_to_split=trial.suggest_float("min_gain_to_split", 0, 5),
                    max_depth=trial.suggest_int("max_depth", 2, 6),
                    bagging_fraction=trial.suggest_float("bagging_fraction", 0.1, 1.0),
                    bagging_freq=trial.suggest_int("bagging_freq", 1, 10),
                    num_iterations=trial.suggest_int("num_iterations", 1, 100),
                    objective="binary",
                    metric="binary_logloss",
                    verbose=-1,
                    num_threads=settings.parallel_processes.num_threads_lightgbm,
                    device_type=settings.parallel_processes.device_type_lightgbm,
                    tree_learner=settings.parallel_processes.tree_learner,
                    force_col_wise="true",
                    seed=42,
                )
            )
            # num_leaves must be smaller than 2^max_depth
            max_num_leaves = 2 ** hyperparameter_dict["max_depth"] - 1
            if max_num_leaves < 90:
                hyperparameter_dict["num_leaves"] = trial.suggest_int(
                    "num_leaves", 2, max_num_leaves
                )
            else:
                hyperparameter_dict["num_leaves"] = trial.suggest_int(
                    "num_leaves", 2, 90
                )
        else:
            raise ValueError(f"Method {selection_method.__name__} is not implemented")

        preprocessed_data_inner_cv_list = (
            preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_iteration]
        )
        scores_list = []
        coefficients_lists = []
        shap_values_lists = []

        # cross validation for the optimization of alpha
        for data_inner_cv_iteration in preprocessed_data_inner_cv_list:
            score, coefficients_list, shap_list = selection_method.calculate_score(
                data_inner_cv_iteration, hyperparameter_dict
            )
            coefficients_lists.append(coefficients_list)
            scores_list.append(score)
            shap_values_lists.append(shap_list)

        assert (
            len(coefficients_lists)
            == len(scores_list)
            == len(shap_values_lists)
            == len(preprocessed_data_inner_cv_list)
        )
        trial.set_user_attr("shap_values", shap_values_lists)
        trial.set_user_attr("macro_feature_importances", coefficients_lists)
        return np.mean(scores_list)

    study = optuna.create_study(
        # storage = "sqlite:///optuna_test.db",
        # load_if_exists = True,
        study_name=f"standard_outer_cv_iteration_{outer_cv_iteration}",
        direction=direction,
        sampler=TPESampler(seed=42),
    )
    if not settings.logging.optuna_trials:  # deactivate logging on cluster
        optuna.logging.set_verbosity(optuna.logging.ERROR)

    study.optimize(
        optuna_objective,
        n_trials=n_trials,
        n_jobs=settings.parallel_processes.hpo_standard,
    )
    return _set_results(preprocessed_data, outer_cv_iteration, selection_method, study)
