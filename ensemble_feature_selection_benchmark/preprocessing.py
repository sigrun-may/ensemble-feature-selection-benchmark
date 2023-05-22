# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT


"""Data Preprocessing."""

import sys
from abc import ABC, abstractmethod
from datetime import datetime
from importlib.machinery import SourceFileLoader
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import skew, yeojohnson
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import PowerTransformer, StandardScaler

from config import settings
from ensemble_feature_selection_benchmark.data_types import (
    DataSplit,
    PreprocessedData,
    TrainTestSplit,
)


if settings["preprocessing"]["yeo_johnson"] == "YeoJohnsonC":
    print(settings.path_yeo_johnson_module)
    yeo_johnson_c = SourceFileLoader(
        "c_accesspoint", settings["path_yeo_johnson_module"]
    ).load_module()


def str_to_class(class_name):
    """Instantiate object from given class name string.

        Args:
            class_name: Name of class to instantiate.

        Returns:
            Object of type class_name.

        """
    print(class_name)
    return getattr(sys.modules[__name__], class_name)


class PowerTransformerBaseClass(ABC):
    @staticmethod
    @abstractmethod
    def transform_train_test_split(
        data_df: pd.DataFrame, train_index, test_index
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass


class CorrelationMatrixCalculatorBaseClass(ABC):
    @staticmethod
    @abstractmethod
    def calculate_correlation_matrix(data_df: pd.DataFrame) -> pd.DataFrame:
        pass


class PreprocessingBaseClass(ABC):
    def __init__(self, _power_transformer, _correlation_matrix_calculator):
        self.power_transformer = _power_transformer
        self.correlation_matrix_calculator = _correlation_matrix_calculator

    @staticmethod
    @abstractmethod
    def get_preprocessed_data_splits(
        self,
        data: pd.DataFrame,
    ) -> PreprocessedData:
        pass

    @staticmethod
    def preprocess_data_split(self, train_index, test_index, data_df) -> DataSplit:
        if settings["preprocessing"]["scale_and_power_transform"]:
            # remove label for transformation
            # labels must be in col 0 (first col)
            unlabeled_data_df = data_df.iloc[:, 1:]
            label = data_df.values[:, 0]

            assert self.power_transformer is not None
            train_np, test_np = self.power_transformer.transform_train_test_split(
                train_index=train_index,
                test_index=test_index,
                unlabeled_data_df=unlabeled_data_df,
            )
            assert train_np.shape[0] == train_index.size, (
                train_np.shape[0],
                train_index.size,
            )
            assert test_np.shape[0] == test_index.size, (
                test_np.shape[0],
                test_index.size,
            )
            assert train_np.shape[1] == len(unlabeled_data_df.columns)
            assert test_np.shape[1] == len(unlabeled_data_df.columns)

            # convert back to labeled dataframe
            test_df = pd.DataFrame(test_np, columns=unlabeled_data_df.columns)

            # calculate correlation matrix for train data
            train_df = pd.DataFrame(train_np, columns=unlabeled_data_df.columns)

            # add label to transformed data
            train_df.insert(0, "label", label[train_index])
            test_df.insert(0, "label", label[test_index])
        else:
            train_df = data_df.iloc[train_index, :]
            test_df = data_df.iloc[test_index, :]

        if settings["preprocessing"]["train_correlation_method"]:
            assert self.correlation_matrix_calculator is not None
            train_correlation_matrix = (
                self.correlation_matrix_calculator.calculate_correlation_matrix(
                    train_df
                )
            )
            assert (
                train_correlation_matrix.shape[1]
                == train_correlation_matrix.shape[0]
                == len(train_df.columns) - 1
            )  # exclude label
        else:
            train_correlation_matrix = None

        assert train_df.shape == (len(train_index), data_df.shape[1])
        assert test_df.shape == (len(test_index), data_df.shape[1])
        assert not train_df.isnull().values.any()
        assert not test_df.isnull().values.any()
        assert len(train_df) > len(test_df)

        return DataSplit(
            train_df,
            test_df,
            train_correlation_matrix,
        )


class PandasPearsonCorrelation(CorrelationMatrixCalculatorBaseClass):
    @staticmethod
    def calculate_correlation_matrix(data_df: pd.DataFrame) -> pd.DataFrame:
        return data_df.iloc[:, 1:].corr(method="pearson")


class PreprocessingSerial(PreprocessingBaseClass):
    power_transformer = None
    correlation_matrix_calculator = None

    @staticmethod
    def get_preprocessed_data_splits(
        self,
        data_df: pd.DataFrame,
    ) -> PreprocessedData:
        outer_preprocessed_data_splits = []
        inner_preprocessed_data_splits_list = []

        # preprocess data splits for outer cv
        k_fold_outer = StratifiedKFold(
            n_splits=settings["cv"]["n_outer_folds"], shuffle=True, random_state=42
        )
        for outer_train_index, test_index in k_fold_outer.split(
            data_df, data_df["label"]
        ):
            outer_preprocessed_data_splits.append(
                self.preprocess_data_split(
                    self,
                    train_index=outer_train_index,
                    test_index=test_index,
                    data_df=data_df,
                )
            )

            # preprocess data splits for inner cv
            inner_preprocessed_data_splits = []

            k_fold_inner = StratifiedKFold(
                n_splits=settings["cv"]["n_inner_folds"], shuffle=True, random_state=42
            )

            remain_df = data_df.iloc[outer_train_index, :]
            assert remain_df.shape[0] == outer_train_index.size
            for inner_train_index, validation_index in k_fold_inner.split(
                remain_df, remain_df["label"]
            ):
                inner_preprocessed_data_splits.append(
                    self.preprocess_data_split(
                        self,
                        train_index=inner_train_index,
                        test_index=validation_index,
                        data_df=remain_df,
                    )
                )
            assert (
                len(inner_preprocessed_data_splits) == settings["cv"]["n_inner_folds"]
            )
            inner_preprocessed_data_splits_list.append(inner_preprocessed_data_splits)
        assert (
            len(inner_preprocessed_data_splits_list)
            == len(outer_preprocessed_data_splits)
            == settings["cv"]["n_outer_folds"]
        )

        return PreprocessedData(
            inner_preprocessed_data_splits_list=inner_preprocessed_data_splits_list,
            outer_preprocessed_data_splits=outer_preprocessed_data_splits,
        )


class YeoJohnsonSklearn(PowerTransformerBaseClass):
    @staticmethod
    def transform_train_test_split(
        unlabeled_data_df: pd.DataFrame, train_index: np.ndarray, test_index: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # check if data is unlabeled
        assert "label" not in unlabeled_data_df.columns

        # transform and standardize test and train data
        power_transformer = PowerTransformer(
            copy=True, method="yeo-johnson", standardize=True
        )
        train_np = power_transformer.fit_transform(
            unlabeled_data_df.iloc[train_index, :]
        )
        test_np = power_transformer.transform(unlabeled_data_df.iloc[test_index, :])

        assert test_np.shape == (len(test_index), unlabeled_data_df.shape[1])
        assert train_np.shape == (len(train_index), unlabeled_data_df.shape[1])

        return TrainTestSplit(train_np=train_np, test_np=test_np)


class YeoJohnsonC(PowerTransformerBaseClass):
    @staticmethod
    def transform_train_test_split(
        unlabeled_data_df: pd.DataFrame, train_index: np.ndarray, test_index: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # check if data is unlabeled
        assert "label" not in unlabeled_data_df.columns
        train_np = unlabeled_data_df.iloc[train_index, :].to_numpy()

        # transform and standardize train data
        (
            transformed_train_np,
            lambdas,
            skews,
            _,
        ) = yeo_johnson_c.automated_yeo_johnson_power_transformation(
            path_to_c_library=settings["path_yeo_johnson_c_library"],
            unlabeled_data_np=train_np,
        )
        # transform test data
        test_pd = unlabeled_data_df.iloc[test_index, :]
        transformed_test_np = np.zeros_like(test_pd.values)

        transformed_test_np_stats = np.zeros_like(test_pd.values)
        skews_c = []
        skews_scipy = []

        counter = 0
        for i, (column_name, test_column) in enumerate(test_pd.items()):
            transformed_test_np[:, i] = yeojohnson(test_column.values, lambdas[i])
            skews_c.append(skew(transformed_test_np[:, i]))
            # _, scipy_optimized_lambda = yeojohnson(train_np[:, i])
            # transformed_test_np_stats[:, i] = yeojohnson(
            #     test_column.values, scipy_optimized_lambda
            # )
        #     skews_scipy.append(skew(transformed_test_np_stats[:, i]))
        #     if abs(skews_scipy[i]) < abs(skews_c[i]):
        #         print(
        #             f"lambda scipy: {scipy_optimized_lambda}, lambda yj c: {lambdas[i]}"
        #         )
        #         print(f"skew scipy: {skews_scipy[i]}, skew yj c: {skews_c[i]}")
        #         # print("Abweichung skew yj c von scipy in Prozent: ",
        #         #        (abs(skews_c[i]) - abs(skews_scipy[i])) / (abs(skews_scipy[i]) * 100))
        #         print("-------")
        #         counter += 1
        # print("-------")
        # print(
        #     f"{counter} von {test_pd.shape[1]} Absolutwerten der skews waren größer bei yj c"
        # )

        # standardize test data
        scaler = StandardScaler()
        scaler.fit(unlabeled_data_df.iloc[train_index, :].to_numpy())
        transformed_test_np = scaler.transform(transformed_test_np, copy=False)
        # transformed_test_np_stats = scaler.transform(
        #     transformed_test_np_stats, copy=False
        # )
        #
        # np.testing.assert_almost_equal(
        #     transformed_test_np, transformed_test_np_stats, decimal=1
        # )
        assert (
            len(lambdas)
            == transformed_train_np.shape[1]
            == transformed_test_np.shape[1]
            == unlabeled_data_df.shape[1]
            == test_pd.shape[1]
        )
        assert transformed_test_np.shape[0] == len(test_index)
        assert transformed_train_np.shape[0] == len(train_index)

        return TrainTestSplit(
            train_np=transformed_train_np, test_np=transformed_test_np
        )


def adapt_data_shape(data_df: pd.DataFrame) -> pd.DataFrame:
    # adapt the data shape
    if settings["data"]["number_of_features"] < data_df.shape[1]:
        data_df = data_df.iloc[:, : settings["data"]["number_of_features"]]
        assert len(data_df.columns) == settings["data"]["number_of_features"]
    if settings["data"]["number_of_samples"] < data_df.shape[0]:
        data_df = data_df.iloc[: settings["data"]["number_of_samples"], :]
        assert len(data_df.shape[0]) == settings["data"]["number_of_samples"]
    print("adapted data shape:", data_df.shape)
    return data_df


def prepare_data(data_df: pd.DataFrame) -> pd.DataFrame:
    # check for missing values
    assert not data_df.isnull().values.any(), "Missing values" + data_df.head()

    # adapt the data shape
    adapted_data_df = adapt_data_shape(data_df)

    #  TODO check for perfectly separated features...

    return adapted_data_df


def preprocess_data() -> PreprocessedData:
    # load pickled preprocessed data
    # # preprocessed_data_id: PreprocessedData|None = load_preprocessed_data()
    # if preprocessed_data_id:
    #     return preprocessed_data_id

    # load data
    input_data_path = f"{settings['cwd_path']}/{settings['data']['folder']}/{settings['data']['name']}/{settings['data']['name']}.csv"
    data_df = pd.read_csv(input_data_path)
    print("data shape:", data_df.shape)
    assert data_df.columns[0] == "label"

    data_df["label"] = data_df["label"]
    data_df = prepare_data(data_df)

    if not settings["preprocessing"]["scale_and_power_transform"]:
        #  TODO daten umkopieren aber nicht verarbeiten
        preprocessor = str_to_class(
            settings["preprocessing"]["preprocessing_parallel"]
        )(
            _power_transformer=None,
            _correlation_matrix_calculator=None,
        )
        preprocessed_data = preprocessor.get_preprocessed_data_splits(
            preprocessor, data_df
        )
    else:
        # data preprocessing
        start_preprocessing = datetime.now()
        preprocessor = str_to_class(
            settings["preprocessing"]["preprocessing_parallel"]
        )(
            _power_transformer=str_to_class(settings["preprocessing"]["yeo_johnson"]),
            _correlation_matrix_calculator=str_to_class(
                settings["preprocessing"]["train_correlation_method"]
            ),
        )
        preprocessed_data = preprocessor.get_preprocessed_data_splits(
            preprocessor, data_df
        )
        preprocessing_time = datetime.now() - start_preprocessing
        print("finished preprocessing in", preprocessing_time)
    return preprocessed_data