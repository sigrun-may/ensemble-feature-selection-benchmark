# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Embedded feature selection with ray tune hyperparameter optimization."""
import os

import math

import numpy as np
import ray
from ray import air, tune
from ray.air.config import RunConfig
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from optuna.samplers import TPESampler


def get_results(
    settings,
    preprocessed_data_list,
    n_trials,
    direction,
    selection_method,
    boosting_type=None,
):
    # define an objective function
    def _objective(config):
        scores_list = []
        coefficients_lists = []
        shap_values_lists = []

        # cross-validation
        for data_split in preprocessed_data_list:
            score, coefficients, shap_list = selection_method.calculate_score(
                ray.get(data_split), config
            )
            scores_list.append(score)
            coefficients_lists.append(coefficients)
            shap_values_lists.append(shap_list)
        return {
            "score": np.mean(scores_list),
            "macro_feature_importances": coefficients_lists,
            "shap_values": shap_values_lists,
        }

    tpe_sampler = TPESampler()
    optuna_search = OptunaSearch(sampler=tpe_sampler, mode=direction, metric="score")
    # optuna_search = ConcurrencyLimiter(optuna_search, max_concurrent=20)

    if "lasso" in selection_method.__name__:
        param_space = {"alpha": tune.qloguniform(0.001, 1, 0.001), "random_state": 42}
    elif "random_forest" in selection_method.__name__:
        param_space = {"n_estimators": tune.randint(1, 150), "random_state": 42}
    elif "svm" in selection_method.__name__:
        param_space = {"C": tune.uniform(1, 10), "random_state": 42}
    elif "lightgbm" in selection_method.__name__:
        if boosting_type == "extra_trees":
            param_space = dict(extra_trees=True, boosting_type="random_forest")
        else:
            param_space = dict(extra_trees=False, boosting_type=boosting_type)
        param_space.update(
            {
                "objective": "binary",
                "bagging_fraction": tune.uniform(0.1, 1.0),
                "bagging_freq": tune.randint(1, 10),
                "max_depth": tune.randint(2, 6),
                # num_leaves must be greater than 2^max_depth
                "num_leaves": tune.randint(2, 90),
                "lambda_l1": tune.uniform(0, 3),
                "min_gain_to_split": tune.uniform(0, 5),
                "min_data_in_leaf": tune.randint(
                    2, math.floor(settings.data.number_of_samples / 2)
                ),
                "num_iterations": tune.randint(1, 100),
                "num_threads": settings.parallel_processes.num_threads_lightgbm,
                "device_type": settings.parallel_processes.device_type_lightgbm,
                "tree_learner": settings.parallel_processes.tree_learner,
                "force_col_wise": True,
                "verbose": -1,
                "seed": 42,
            }
        )
    else:
        raise ValueError(f"Method {selection_method.__name__} is not implemented")

    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    # start a Tune run
    tuner = tune.Tuner(
        _objective,
        param_space=param_space,
        tune_config=tune.TuneConfig(
            num_samples=n_trials,
            max_concurrent_trials=settings.parallel_processes.max_concurrent_trials_hpo_ray,
            search_alg=optuna_search,
        ),
        run_config=RunConfig(
            local_dir=f"{settings.cwd_path}/ray",
            verbose=0,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=0, checkpoint_at_end=False
            ),
            sync_config=tune.SyncConfig(syncer='auto'),
            log_to_file=False,
        ),
    )
    return tuner.fit()


def select_features(
    settings,
    preprocessed_data,
    outer_cv_loop,
    n_trials,
    direction,
    selection_method,
    boosting_type=None,
) -> dict:
    if isinstance(preprocessed_data, ray._raylet.ObjectRef):
        preprocessed_data = ray.get(preprocessed_data)
    if isinstance(
        preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_loop],
        ray._raylet.ObjectRef,
    ):
        inner_preprocessed_data_splits_list = ray.get(
            preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_loop]
        )
    else:
        inner_preprocessed_data_splits_list = (
            preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_loop]
        )
    results = get_results(
        settings,
        inner_preprocessed_data_splits_list,
        n_trials,
        direction,
        selection_method,
        boosting_type,
    )
    best_result = results.get_best_result(metric="score", mode="max")
    del results
    del inner_preprocessed_data_splits_list

    # build model for micro_feature_importance
    train_data_outer_cv_df = ray.get(
        preprocessed_data.outer_preprocessed_data_splits[outer_cv_loop]
    ).train_data_outer_cv_df
    micro_feature_importance = selection_method.calculate_micro_feature_importance(
        train_data_outer_cv_df, best_result.config
    )

    if "shap_values" in best_result.metrics.keys():
        result_dict = dict(shap_values=best_result.metrics["shap_values"])
    else:
        result_dict = dict(shap_values=[])

    result_dict["validation_metric_value"] = best_result.metrics["score"]
    result_dict["best_hyper_parameters"] = best_result.config
    result_dict["macro_feature_importances"] = best_result.metrics[
        "macro_feature_importances"
    ]
    result_dict["micro_feature_importance"] = list(micro_feature_importance)
    del best_result
    assert len(result_dict["macro_feature_importances"]) == len(
        preprocessed_data.inner_preprocessed_data_splits_list[outer_cv_loop]
    )
    return result_dict
