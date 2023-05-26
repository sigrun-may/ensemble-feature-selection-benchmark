# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Data types for preprocessed input data."""

from collections import namedtuple
from dataclasses import dataclass
from typing import List


DataSplit = namedtuple(
    "DataSplit",
    ["train_data_outer_cv_df", "validation_df", "train_correlation_matrix_df"],
)
TrainTestSplit = namedtuple("TrainTestSplit", ["train_np", "test_np"])


@dataclass
class PreprocessedData:
    """Class to hold the complete preprocessed data.

    Attributes:
        inner_preprocessed_data_splits_list:
        outer_preprocessed_data_splits:

    """

    inner_preprocessed_data_splits_list: List[List[DataSplit]]
    outer_preprocessed_data_splits: List[DataSplit]

    def __eq__(self, other):
        return (
            self.inner_preprocessed_data_splits_list
            == other.inner_preprocessed_data_splits_list
        ) and (
            self.outer_preprocessed_data_splits == other.outer_preprocessed_data_splits
        )
