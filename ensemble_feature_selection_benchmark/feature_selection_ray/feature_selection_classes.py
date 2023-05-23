# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Feature Selection methods for ensemble feature selection."""

import logging
import sys
from abc import ABC, abstractmethod

from ensemble_feature_selection_benchmark.data_types import PreprocessedData
from ensemble_feature_selection_benchmark.feature_selection_methods import (
    standard_lightgbm,
    standard_sklearn_lasso,
    standard_sklearn_random_forest,
    standard_sklearn_svm,
)
from ensemble_feature_selection_benchmark.feature_selection_ray import (
    embedded_feature_selection_optuna,
    embedded_feature_selection_hpo_ray,
    reverse_selection,
)


_logger = logging.getLogger(__name__)


def str_to_class(class_name):
    return getattr(sys.modules[__name__], class_name)


class FeatureSelectionBaseClass(ABC):
    @staticmethod
    @abstractmethod
    def select_feature_subsets(
        data: PreprocessedData, outer_cv_iteration: int, **kwargs
    ):
        pass


class LassoSklearn(FeatureSelectionBaseClass):
    @staticmethod
    def select_feature_subsets(data, outer_cv_iteration: int, **kwargs):
        assert len(kwargs) == 1
        assert "settings_id" in kwargs
        settings_id = kwargs["settings_id"]

        if settings_id.parallel_processes.max_concurrent_trials_hpo_ray != 1:
            print(
                "max_concurrent_trials_hpo_ray",
                settings_id.parallel_processes.max_concurrent_trials_hpo_ray,
            )
            return embedded_feature_selection_hpo_ray.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.LassoSklearnOptuna.n_trials,
                direction="max",  # maximizing r2
                selection_method=standard_sklearn_lasso,
            )
        else:
            return embedded_feature_selection_optuna.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.LassoSklearnOptuna.n_trials,
                direction="maximize",  # maximizing r2
                selection_method=standard_sklearn_lasso,
            )


# check C lasso https://github.com/Leo-Simpson/c-lasso as possibly faster method


class RandomForestSklearn(FeatureSelectionBaseClass):
    @staticmethod
    def select_feature_subsets(data, outer_cv_iteration: int, **kwargs):
        assert len(kwargs) == 1
        assert "settings_id" in kwargs
        settings_id = kwargs["settings_id"]

        if settings_id.parallel_processes.max_concurrent_trials_hpo_ray != 1:
            return embedded_feature_selection_hpo_ray.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.RandomForestSklearnOptuna.n_trials,
                direction="max",  # maximizing accuracy
                selection_method=standard_sklearn_random_forest,
            )
        else:
            return embedded_feature_selection_optuna.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.RandomForestSklearnOptuna.n_trials,
                direction="maximize",  # maximizing accuracy
                selection_method=standard_sklearn_random_forest,
            )


class SVC(FeatureSelectionBaseClass):
    @staticmethod
    def select_feature_subsets(data, outer_cv_iteration: int, **kwargs):
        assert len(kwargs) == 1
        assert "settings_id" in kwargs
        settings_id = kwargs["settings_id"]

        if settings_id.parallel_processes.max_concurrent_trials_hpo_ray != 1:
            return embedded_feature_selection_hpo_ray.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.SvcSklearnOptuna.n_trials,
                direction="max",  # maximizing accuracy
                selection_method=standard_sklearn_svm,
            )
        else:
            return embedded_feature_selection_optuna.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.SvcSklearnOptuna.n_trials,
                direction="maximize",  # maximizing accuracy
                selection_method=standard_sklearn_svm,
            )


class RandomForestLightGBM(FeatureSelectionBaseClass):
    """Lightgbm random forest feature selection."""

    @staticmethod
    def select_feature_subsets(data, outer_cv_iteration: int, **kwargs):
        """Select subset of possibly relevant features with lightgbm random forest.

        Args:
            data: Preprocessed yeo johnson transformed data and corresponding correlation matrices.
            outer_cv_iteration: Index of outer cross-validation loop.
            **kwargs: settings_id -> ray object id for dynaconf settings

        Returns:
            Selected feature subsets (micro/ macro coefficients and shap values).

        """
        assert len(kwargs) == 1
        assert "settings_id" in kwargs
        settings_id = kwargs["settings_id"]

        if settings_id.parallel_processes.max_concurrent_trials_hpo_ray != 1:
            return embedded_feature_selection_hpo_ray.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.LightgbmOptuna.n_trials,
                direction="min",  # minimizing log loss
                selection_method=standard_lightgbm,
                boosting_type="random_forest",
            )
        else:
            return embedded_feature_selection_optuna.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.LightgbmOptuna.n_trials,
                direction="minimize",  # minimizing log loss
                selection_method=standard_lightgbm,
                boosting_type="random_forest",
            )


class GradientBoostingDecisionTreeLightGBM(FeatureSelectionBaseClass):
    """Lightgbm feature selection."""

    @staticmethod
    def select_feature_subsets(data, outer_cv_iteration: int, **kwargs):
        """Select subset of possibly relevant features with lightgbm.

        Args:
            data: Preprocessed yeo johnson transformed data and corresponding correlation matrices.
            outer_cv_iteration: Index of outer cross-validation loop.
            **kwargs: settings_id -> ray object id for dynaconf settings

        Returns:
            Selected feature subsets (micro/ macro coefficients and shap values).

        """
        assert len(kwargs) == 1
        assert "settings_id" in kwargs
        settings_id = kwargs["settings_id"]

        if settings_id.parallel_processes.max_concurrent_trials_hpo_ray != 1:
            return embedded_feature_selection_hpo_ray.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.LightgbmOptuna.n_trials,
                direction="min",  # minimizing log loss
                selection_method=standard_lightgbm,
                boosting_type="gbdt",
            )
        else:
            return embedded_feature_selection_optuna.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.LightgbmOptuna.n_trials,
                direction="minimize",  # minimizing log loss
                selection_method=standard_lightgbm,
                boosting_type="gbdt",
            )


class ExtraTreesLightGBM(FeatureSelectionBaseClass):
    """Lightgbm extra trees feature selection."""

    @staticmethod
    def select_feature_subsets(data, outer_cv_iteration: int, **kwargs):
        """Select subset of possibly relevant features with lightgbm extra trees.

        Args:
            data: Preprocessed yeo johnson transformed data and corresponding correlation matrices.
            outer_cv_iteration: Index of outer cross-validation loop.
            **kwargs: settings_id -> ray object id for dynaconf settings

        Returns:
            Selected feature subsets (micro/ macro coefficients and shap values).

        """
        assert len(kwargs) == 1
        assert "settings_id" in kwargs
        settings_id = kwargs["settings_id"]

        if settings_id.parallel_processes.max_concurrent_trials_hpo_ray != 1:
            return embedded_feature_selection_hpo_ray.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.LightgbmOptuna.n_trials,
                direction="min",  # minimizing log loss
                selection_method=standard_lightgbm,
                boosting_type="extra_trees",
            )
        else:
            return embedded_feature_selection_optuna.select_features(
                settings_id,
                data,
                outer_cv_iteration,
                n_trials=settings_id.LightgbmOptuna.n_trials,
                direction="minimize",  # minimizing log loss
                selection_method=standard_lightgbm,
                boosting_type="extra_trees",
            )


# class LightGBM(FeatureSelectionBaseClass):
#     @staticmethod
#     def select_feature_subsets(data, outer_cv_iteration: int, **kwargs):
#         assert len(kwargs) == 1
#         assert "settings_id" in kwargs
#         settings_id = kwargs["settings_id"]
#
#         if settings_id.parallel_processes.max_concurrent_trials_hpo_ray != 1:
#             return embedded_feature_selection_hpo_ray.select_features(
#                 settings_id,
#                 data,
#                 outer_cv_iteration,
#                 n_trials=settings_id.LightgbmOptuna.n_trials,
#                 direction="max",  # maximizing accuracy
#                 selection_method=standard_lightgbm,
#             )
#         else:
#             return embedded_feature_selection_optuna.select_features(
#                 settings_id,
#                 data,
#                 outer_cv_iteration,
#                 n_trials=settings_id.LightgbmOptuna.n_trials,
#                 direction="minimize",  # minimizing log loss
#                 selection_method=standard_lightgbm,
#             )


class ReverseLassoSklearn(FeatureSelectionBaseClass):
    @staticmethod
    def select_feature_subsets(data, outer_cv_iteration: int, **kwargs):
        assert len(kwargs) == 1
        assert "settings_id" in kwargs
        settings_id = kwargs["settings_id"]

        return reverse_selection.calculate_labeled_and_unlabeled_validation_metrics(
            settings_id,
            preprocessed_data_id=data,
            selection_method=standard_sklearn_lasso.train_lasso_sklearn,
            outer_cv_iteration=outer_cv_iteration,
        )


# class HiLassoOptuna(FeatureSelectionBaseClass):
#     @staticmethod
#     def select_feature_subsets(settings, data: PreprocessedData, outer_cv_iteration: int, target="label"):
#         pass
#
#
# class RelaxedLassoOptuna(FeatureSelectionBaseClass):
#     @staticmethod
#     def select_feature_subsets(settings, data: PreprocessedData, outer_cv_iteration: int, target="label"):
#         return lasso_sklearn_optuna.select_features(data, target, 0, lasso_methods.train_relaxed_lasso_optuna)
