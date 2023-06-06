# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Reverse feature selection for high-dimensional data with tiny sample size."""

import logging
import math
import numpy as np
from typing import List

import optuna
import pandas as pd
import ray
from optuna import TrialPruned
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score
from more_itertools import chunked

_logger = logging.getLogger(__name__)


def _get_train_validation_data(settings, data_split, target_feature, labeled):
    train_df, validation_df, train_correlation_matrix_df = ray.get(data_split)
    # remove correlations to target feature
    uncorrelated_features_mask = (
        train_correlation_matrix_df[target_feature]
        .abs()
        .le(
            settings.reverse_fs_parameter.train_correlation_threshold,
            axis="index",
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
        settings,
        preprocessed_data,
        outer_cv_iteration,
        target_feature,
        selection_method,
        labeled,
        hyperparameters,
):
    if isinstance(preprocessed_data, ray._raylet.ObjectRef):
        inner_preprocessed_data_splits_list = ray.get(
            preprocessed_data
        ).inner_preprocessed_data_splits_list[outer_cv_iteration]
        assert isinstance(inner_preprocessed_data_splits_list, List)
        for list_element in inner_preprocessed_data_splits_list:
            assert isinstance(list_element, ray._raylet.ObjectRef)
    else:
        inner_preprocessed_data_splits_list = (
            preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_iteration]
        )
    predicted_y = []
    true_y = []
    # cross validation
    for data_split in inner_preprocessed_data_splits_list:
        x_train, y_train, x_validation, y_validation = _get_train_validation_data(
            settings, data_split, target_feature, labeled
        )
        model = selection_method(x_train, y_train, parameters=hyperparameters)
        # terminate trial, if label was not selected for reverse feature selection
        if labeled and math.isclose(model.coef_[x_train.columns.get_loc("label")], 0):
            raise TrialPruned

        predicted_y_validation = model.predict(x_validation)
        predicted_y.extend(predicted_y_validation)
        true_y.extend(y_validation)
    return r2_score(true_y, predicted_y)


def _optimize_evaluation_metric(
        settings, preprocessed_data, outer_cv_iteration, target_feature, selection_method
):
    def optuna_objective(trial):
        """Optimize regularization parameter alpha for lasso regression."""

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
        else:
            raise ValueError("No valid selection method for reverse feature selection")
        return _calculate_validation_metric(
            settings,
            preprocessed_data,
            outer_cv_iteration,
            target_feature,
            selection_method,
            labeled=True,
            hyperparameters=hyperparameter_dict,
        )

    study_name = f"lasso_{target_feature}_iteration_{outer_cv_iteration}"
    study = optuna.create_study(
        # storage=settings.data_storage.path_sqlite_for_optuna,
        # load_if_exists = True,
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(
            n_startup_trials=3,
        ),
    )
    if not settings.logging.optuna_trials:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
    study.optimize(
        optuna_objective,
        n_trials=settings.reverse_fs_parameter.n_trials,
        n_jobs=settings.parallel_processes.hpo_reverse,
        gc_after_trial=True,
    )
    # initialize return values
    best_trial_value = 0
    best_params = {}
    # calculate unlabeled metric for comparison, of labeled result is promising
    if len(study.best_trials) > 0 and study.best_trial.value > 0:
        if isinstance(preprocessed_data, ray._raylet.ObjectRef):
            outer_preprocessed_data_split = ray.get(
                preprocessed_data
            ).outer_preprocessed_data_splits[outer_cv_iteration]
        else:
            outer_preprocessed_data_split = (
                preprocessed_data.outer_preprocessed_data_splits[outer_cv_iteration]
            )
        x_remain, y_remain, _, _ = _get_train_validation_data(
            settings,
            outer_preprocessed_data_split,
            target_feature,
            labeled=True,
        )
        assert x_remain.columns[0] == "label"
        # build LASSO model for micro_feature_selection
        lasso = selection_method(x_remain, y_remain, parameters=study.best_params)
        if lasso.coef_[0] != 0:
            best_trial_value = study.best_trial.value
            best_params = study.best_params
    # optuna.delete_study(study_name=study_name, storage=settings.data_storage.path_sqlite_for_optuna)
    return best_trial_value, best_params


def calculate_labeled_and_unlabeled_validation_metrics(
        settings_id, preprocessed_data_id, selection_method, outer_cv_iteration
) -> pd.DataFrame:
    if isinstance(preprocessed_data_id, ray._raylet.ObjectRef):
        preprocessed_data_id = ray.get(preprocessed_data_id)
    if isinstance(
            preprocessed_data_id.outer_preprocessed_data_splits[0], ray._raylet.ObjectRef
    ):
        labeled_feature_names = ray.get(
            preprocessed_data_id.outer_preprocessed_data_splits[0]
        ).train_data_outer_cv_df.columns
    else:
        labeled_feature_names = preprocessed_data_id.outer_preprocessed_data_splits[
            0
        ].train_data_outer_cv_df.columns
    #     assert isinstance(
    #         preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_iteration][0], ray._raylet.ObjectRef
    #     )
    #     labeled_feature_names = ray.get(
    #         preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_iteration][0]
    #     ).train_df.columns
    #     del preprocessed_data
    # elif isinstance(
    #     preprocessed_data_id.inner_preprocessed_data_splits_list[outer_cv_iteration][0], ray._raylet.ObjectRef
    # ):
    #     labeled_feature_names = ray.get(
    #         preprocessed_data_id.inner_preprocessed_data_splits_list[outer_cv_iteration][0]
    #     ).train_df.columns
    # else:
    #     labeled_feature_names = preprocessed_data_id.inner_preprocessed_data_splits_list[outer_cv_iteration][
    #         0
    #     ].train_df.columns
    feature_names = labeled_feature_names[1:]  # exclude label as target
    del labeled_feature_names
    labeled_validation_metrics = []
    unlabeled_validation_metrics = []

    for batch in chunked(feature_names, settings_id.parallel_processes.reverse_feature_selection):
        labeled_validation_metrics_chunk = []
        unlabeled_validation_metrics_chunk = []

        for target_feature in batch:
            if settings_id.parallel_processes.reverse_feature_selection > 1:
                (
                    labeled,
                    unlabeled,
                ) = _remote_calculate_validation_metrics_per_feature(
                    # ) = _remote_calculate_validation_metrics_per_feature.options(
                    #     memory=0.5 * 1024 * 1024 * 1024
                    # ).remote(
                    settings_id,
                    preprocessed_data_id,
                    outer_cv_iteration,
                    target_feature,
                    selection_method,
                )
            else:
                labeled, unlabeled = _calculate_validation_metrics_per_feature(
                    settings_id,
                    preprocessed_data_id,
                    outer_cv_iteration,
                    target_feature,
                    selection_method,
                )
            labeled_validation_metrics_chunk.append(labeled)
            del labeled
            unlabeled_validation_metrics_chunk.append(unlabeled)
            del unlabeled
            _logger.debug(f"{target_feature} done")

        if settings_id.parallel_processes.reverse_feature_selection > 1:
            # delete object ids from ray to free memory
            loaded_unlabeled_validation_metrics = ray.get(unlabeled_validation_metrics_chunk)
            del unlabeled_validation_metrics_chunk
            unlabeled_validation_metrics.extend(loaded_unlabeled_validation_metrics)
            del loaded_unlabeled_validation_metrics

            loaded_labeled_validation_metrics = ray.get(labeled_validation_metrics_chunk)
            del labeled_validation_metrics_chunk
            labeled_validation_metrics.extend(loaded_labeled_validation_metrics)
            del loaded_labeled_validation_metrics
    assert (
            len(unlabeled_validation_metrics)
            == len(labeled_validation_metrics)
            == len(feature_names)
    )
    for list_element in unlabeled_validation_metrics:
        assert isinstance(list_element, float)
        assert not math.isnan(list_element)
    for list_element in labeled_validation_metrics:
        assert isinstance(list_element, float)
        assert not math.isnan(list_element)
    validation_metrics = pd.DataFrame(
        data=unlabeled_validation_metrics,
        index=feature_names,
        columns=["unlabeled"],
    )
    validation_metrics["labeled"] = labeled_validation_metrics
    return validation_metrics


def _calculate_validation_metrics_per_feature(
        settings, preprocessed_data, outer_cv_iteration, target_feature, selection_method
):
    # optimize hyperparameters with labeled training data
    labeled_validation_metric_value, best_parameters = _optimize_evaluation_metric(
        settings,
        preprocessed_data,
        outer_cv_iteration,
        target_feature,
        selection_method,
    )
    if len(best_parameters) > 0:
        # calculate validation metric for unlabeled training data using the given optimized hyperparameters
        unlabeled_validation_metric_value = _calculate_validation_metric(
            settings,
            preprocessed_data,
            outer_cv_iteration,
            target_feature,
            selection_method,
            labeled=False,
            hyperparameters=best_parameters,
        )
    else:
        unlabeled_validation_metric_value = 0
        assert math.isclose(labeled_validation_metric_value, 0)
        _logger.info("No trial completed")

    return labeled_validation_metric_value, unlabeled_validation_metric_value


@ray.remote(num_returns=2)
def _remote_calculate_validation_metrics_per_feature(
        settings, preprocessed_data, outer_cv_iteration, target_feature, selection_method
):
    return _calculate_validation_metrics_per_feature(
        settings,
        preprocessed_data,
        outer_cv_iteration,
        target_feature,
        selection_method,
    )
