# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Reverse feature selection for high-dimensional data with tiny sample size."""

import logging
import math

import optuna
import pandas as pd
from optuna import TrialPruned
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score

from config import settings
from ensemble_feature_selection_benchmark.data_types import DataSplit


_logger = logging.getLogger(__name__)


def _get_uncorrelated_train_and_validation_data(
    data_split: DataSplit, target_feature, labeled
):
    train_df, validation_df, train_correlation_matrix_df = data_split
    # remove correlations to target feature
    uncorrelated_features_mask = (
        train_correlation_matrix_df[target_feature]
        .abs()
        .le(
            settings.reverse_fs_parameter.train_correlation_threshold,
            axis="index",
            # For a correlation matrix filled only with the lower half,
            # the first elements up to the diagonal would have to be read
            # with axis="index" and the further elements after the diagonal
            # with axis="column".
        )
    )
    uncorrelated_features_index = (
        train_df.iloc[:, 1:].loc[:, uncorrelated_features_mask].columns
    )
    assert "label" not in uncorrelated_features_index
    assert len(uncorrelated_features_index) > 0
    if labeled:
        # insert label
        uncorrelated_features_index = uncorrelated_features_index.insert(0, "label")
        assert "label" in uncorrelated_features_index

    # prepare train data
    x_train = train_df[uncorrelated_features_index]
    # enlarge difference between classes for reverse feature selection
    y_train = train_df[target_feature].values.reshape(-1, 1) * 100

    # prepare validation data
    x_validation = validation_df[uncorrelated_features_index]
    # enlarge difference between classes for reverse feature selection
    y_validation = validation_df[target_feature] * 100

    return x_train, y_train, x_validation, y_validation


def _calculate_validation_metric(
    inner_cv_preprocessed_data_splits,
    target_feature,
    selection_method,
    labeled,
    hyperparameters,
):
    predicted_y = []
    true_y = []
    # cross validation
    for data_split in inner_cv_preprocessed_data_splits:
        (
            x_train,
            y_train,
            x_validation,
            y_validation,
        ) = _get_uncorrelated_train_and_validation_data(
            data_split, target_feature, labeled
        )
        model = selection_method(x_train, y_train, hyperparameters)
        # terminate trial, if label was not selected for reverse feature selection
        if labeled:
            if (
                "lasso" in selection_method.__name__
                and math.isclose(model.coef_[x_train.columns.get_loc("label")], 0.0)
            ) or (
                "andom" in selection_method.__name__
                and math.isclose(
                    model.feature_importances_[x_train.columns.get_loc("label")], 0.0
                )
            ):
                raise TrialPruned

        predicted_y_validation = model.predict(x_validation)
        predicted_y.extend(predicted_y_validation)
        true_y.extend(y_validation)
    return r2_score(true_y, predicted_y)


def _optimize(
    preprocessed_data, outer_cv_iteration, target_feature, selection_method, direction
):
    def _optuna_objective(trial):
        """Optimize hyperparameter."""
        # if optuna_study_pruner.study_patience_pruner(
        #     trial, epsilon=0.001, warm_up_steps=20, patience=5
        # ) or optuna_study_pruner.study_no_improvement_pruner(
        #     trial,
        #     epsilon=0.01,
        #     warm_up_steps=30,
        #     number_of_similar_best_values=5,
        #     threshold=0.1,
        # ):
        #     print("study stopped")
        #     trial.study.stop()
        #     raise TrialPruned()
        if "lasso" in selection_method.__name__:
            hyperparameter_dict = {
                "alpha": trial.suggest_float("alpha", 0.01, 1.0, log=True)
            }
        elif ("random_forest" in selection_method.__name__) or ("trees" in selection_method.__name__):
            hyperparameter_dict = {
                "random_state": 42,
                "criterion": "absolute_error",
                "max_depth": trial.suggest_int("max_depth", 2, 6),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    2,
                    math.floor(settings.data.number_of_samples / 2),
                ),
                "n_jobs": 1,
            }
        else:
            raise ValueError("No valid selection method for reverse feature selection")
        return _calculate_validation_metric(
            preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_iteration],
            target_feature,
            selection_method,
            labeled=True,
            hyperparameters=hyperparameter_dict,
        )

    if "lasso" in selection_method.__name__:
        n_trials = settings.reverse_fs_lasso_parameter.n_trials
        n_startup_trials = settings.reverse_fs_lasso_parameter.n_startup_trials
    else:
        n_trials = settings.reverse_fs_random_forest_parameter.n_trials
        n_startup_trials = settings.reverse_fs_random_forest_parameter.n_startup_trials

    study_name = f"lasso_{target_feature}_iteration_{outer_cv_iteration}"
    study = optuna.create_study(
        # storage=settings.data_storage.path_sqlite_for_optuna,
        # load_if_exists = True,
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=n_startup_trials,
        ),
    )
    if not settings.logging.optuna_trials:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(
        _optuna_objective,
        n_trials=n_trials,
        n_jobs=settings.parallel_processes.hpo_reverse,
    )
    # initialize return values
    best_trial_value = 0
    best_params = {}
    # calculate unlabeled metric for comparison, if labeled result is promising
    if len(study.best_trials) > 0 and study.best_trial.value > 0:
        (
            x_train_outer_cv,
            y_train_outer_cv,
            _,
            _,
        ) = _get_uncorrelated_train_and_validation_data(
            preprocessed_data.outer_preprocessed_data_splits[outer_cv_iteration],
            target_feature,
            labeled=True,
        )
        assert x_train_outer_cv.columns[0] == "label"
        # build model for micro_feature_selection
        model = selection_method(x_train_outer_cv, y_train_outer_cv, study.best_params)
        if (("lasso" in selection_method.__name__) and (model.coef_[0] != 0.0)) or (
            ("lasso" not in selection_method.__name__)
            and (model.feature_importances_[0] != 0.0)
        ):
            best_trial_value = study.best_trial.value
            best_params = study.best_params
    # optuna.delete_study(study_name=study_name, storage=settings.data_storage.path_sqlite_for_optuna)
    return best_trial_value, best_params


def labeled_and_unlabeled_training(
    preprocessed_data, selection_method, direction, outer_cv_iteration
) -> pd.DataFrame:
    """Calculate validation metrics with labeled and unlabeled training data for reverse feature selection.

    Args:
        preprocessed_data: Preprocessed yeo johnson transformed data and corresponding correlation matrices.
        selection_method: Selection method for embedded feature selection.
        direction: direction for optimization -> can be "maximize" or "minimize"
        outer_cv_iteration: Index of outer cross-validation loop.

    Returns:
        Results for labeled and unlabeled training.
    """
    labeled_feature_names = preprocessed_data.outer_preprocessed_data_splits[
        0
    ].train_data_outer_cv_df.columns
    feature_names = labeled_feature_names[1:]  # exclude label as target
    labeled_validation_metrics = []
    unlabeled_validation_metrics = []
    for target_feature in feature_names:
        (
            labeled_validation_metric_value,
            unlabeled_validation_metric_value,
        ) = _calculate_validation_metrics_per_feature(
            preprocessed_data=preprocessed_data,
            outer_cv_iteration=outer_cv_iteration,
            target_feature=target_feature,
            selection_method=selection_method,
            direction=direction,
        )
        labeled_validation_metrics.append(labeled_validation_metric_value)
        unlabeled_validation_metrics.append(unlabeled_validation_metric_value)
        _logger.debug(f"{target_feature} done")
    assert (
        len(unlabeled_validation_metrics)
        == len(labeled_validation_metrics)
        == len(feature_names)
    )
    validation_metrics = pd.DataFrame(
        data=unlabeled_validation_metrics,
        index=feature_names,
        columns=["unlabeled"],
    )
    validation_metrics["labeled"] = labeled_validation_metrics
    return validation_metrics


def _calculate_validation_metrics_per_feature(
    preprocessed_data, outer_cv_iteration, target_feature, selection_method, direction
):
    # optimize hyperparameters with labeled training data
    labeled_validation_metric_value, best_parameters = _optimize(
        preprocessed_data,
        outer_cv_iteration,
        target_feature,
        selection_method,
        direction,
    )
    # if any trial was completed
    if len(best_parameters) > 0:
        # calculate validation metric with cross-validation for unlabeled training data
        # using the given optimized hyperparameters
        unlabeled_validation_metric_value = _calculate_validation_metric(
            preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_iteration],
            target_feature,
            selection_method,
            labeled=False,
            hyperparameters=best_parameters,
        )
    else:
        unlabeled_validation_metric_value = 0.0
        assert math.isclose(labeled_validation_metric_value, 0.0)
        _logger.info("No trial completed")

    return labeled_validation_metric_value, unlabeled_validation_metric_value
