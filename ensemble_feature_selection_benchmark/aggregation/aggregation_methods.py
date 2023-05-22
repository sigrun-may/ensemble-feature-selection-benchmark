# Copyright (c) 2023 Sigrun May, Ostfalia Hochschule für angewandte Wissenschaften
# This software is distributed under the terms of the MIT license
# which is available at https://opensource.org/licenses/MIT

"""Aggregation methods for ensemble feature selection."""

import logging
import statistics
import sys
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import pyrankvote
from pref_voting.iterative_methods import coombs, instant_runoff
from pref_voting.other_methods import bucklin
from pref_voting.profiles import Profile
from pyrankvote import Ballot, Candidate

from config import settings


_logger = logging.getLogger(__name__)


def str_to_class(class_name):
    """Instantiate object from given class name string.

        Args:
            class_name: Name of class to instantiate.

        Returns:
            Object of type class_name.

        """
    return getattr(sys.modules[__name__], class_name)


class AggregationBaseClass(ABC):
    """Base class for aggregation methods"""
    @staticmethod
    @abstractmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Aggregate the results of the single feature selection.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods

        """

        pass


class Sum(AggregationBaseClass):
    """Sum feature selection results"""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Calculate sum the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (sum)

        """

        return data.sum(axis=1)


class SumRanked(AggregationBaseClass):
    """Feature ranking by sum of the results of feature selection methods"""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Calculate the feature rankings by the sum of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (feature rankings by sum)

        """

        return data.sum(axis=1).rank(ascending=False)


class ArithmeticMean(AggregationBaseClass):
    """Arithmetic mean of feature selection methods."""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Calculate the arithmetic mean of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (arithmetic mean)

        """

        return data.mean(axis=1)


class ArithmeticMeanRanked(AggregationBaseClass):
    """Feautre ranking of arithmetic mean of the results of feature selection methods."""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Calculate the feature rankings by the arithmetic mean of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (feature rankings by arithmetic mean)

        """

        arithmetic_means = data.mean(axis=1)
        return (
            arithmetic_means.rank(ascending=False)
            if arithmetic_means.le(1).all().all()
            else arithmetic_means.rank()
        )


class GeometricMean(AggregationBaseClass):  # use only on ranked input datasets!
    """Geometric mean of ranked feature selection results"""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the geometric mean of the ranking results of feature selection methods.

        Args:
            data: ranking results of feature selection methods

        Returns:
            aggregated ranking results of feature selection methods (geometric mean)

        """

        return data.apply((lambda row: statistics.geometric_mean(row)), axis=1)


class GeometricMeanRanked(AggregationBaseClass):  # use only on ranked input datasets!
    """Feautre ranking of geometric mean of ranked feature selection results"""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the feature rankings by the geometric mean of the results of feature selection methods.

        Args:
            data: ranking results of feature selection methods

        Returns:
            aggregated ranking results of feature selection methods (feature ranking by geometric mean)

        """

        return data.apply((lambda row: statistics.geometric_mean(row)), axis=1).rank()


class Median(AggregationBaseClass):
    """Median of feature selection results."""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the median of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (median)

        """

        return data.apply((lambda row: statistics.median(row)), axis=1)


class MedianRanked(AggregationBaseClass):
    """Feature ranking of Median of feature selection results."""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the feature rankings by the median of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (feature ranking by median)

        """

        medians = data.apply((lambda row: statistics.median(row)), axis=1)
        return (
            medians.rank(ascending=False)
            if medians.le(1).all().all()
            else medians.rank()
        )


class Count(AggregationBaseClass):
    """Count the number of times a feature is selected by the feature selection methods"""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Count the number of times a feature is selected by the feature selection
        methods. A feature is considered as selected, if the feature importance is not zero.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (count number of times feature is selected)

        """

        selected_features = data != 0
        return selected_features.sum(axis=1)


class CountRanked(AggregationBaseClass):
    """Feature ranking by the count the number of times a feature is selected by the feature selection methods"""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Calculate feature ranking by the count the number of times a feature is selected by the feature selection
        methods. A feature is considered as selected, if the feature importance is not zero.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (feature ranking by count number of times feature is
            selected)

        """

        selected_features = data != 0
        return selected_features.sum(axis=1).rank(ascending=False)


class Min(AggregationBaseClass):
    """Minimum of the results of the feature selection methods"""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the minimum of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (minimum)

        """

        return data.min(axis=1)


class MinRanked(AggregationBaseClass):
    """Feature ranking by the minimum of the feature selection methods."""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the feature ranking by the minimum of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (feature ranking by minimum)

        """

        minimum = data.min(axis=1)
        return (
            minimum.rank(ascending=False)
            if minimum.le(1).all().all()
            else minimum.rank()
        )


class Max(AggregationBaseClass):
    """Maximum of the feature selection results."""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the maximum of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (maximum)

        """

        return data.max(axis=1)


class MaxRanked(AggregationBaseClass):
    """Feature rankings by the maximum of the feature selection methods"""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the feature rankings by the maximum of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (feature ranking by maximum)

        """

        maximum = data.max(axis=1)
        return (
            maximum.rank(ascending=False)
            if maximum.le(1).all().all()
            else maximum.rank()
        )


class WeightedAveraging(
    AggregationBaseClass
):  # needs a dataframe with an added row called "acc" that contains the accuracy of the pred.model
    """Weighted average of the results of the feature selection methods."""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the weighted average of the results of the feature selection methods. The weights are the
        accuracies of the prediction model.

        Args:
            data: results of feature selection methods, must have a row called "acc" that contains the accuracy of the
                    prediction model

        Returns:
            aggregated results of feature selection methods (weighted average)

        """

        weights = data.loc["acc"]
        weights = weights.apply(lambda weight: weight / sum(weights))
        data = data.drop("acc")
        number_of_fs_methods = len(data.columns)
        weighted_avg = {}
        for feature, value in data.iterrows():
            new_value = 0
            for fs_method, val in value.items():
                new_value = new_value + (val * weights[fs_method])
            weighted_avg[feature] = new_value / number_of_fs_methods
        return pd.DataFrame.from_dict(weighted_avg, orient="index")


class WeightedAveragingRanked(AggregationBaseClass):
    """Feature rankings by the weighted average of the results of the feature selection methods."""
    @staticmethod
    def aggregate(data: pd.DataFrame, **kwargs) -> pd.DataFrame:

        """ Calculate the feature rankings by the weighted average of the results of the feature selection methods.
        The weights are the accuracies of the prediction model.

        Args:
            data: results of feature selection methods, must have a row called "acc" that contains the accuracy of the
                    prediction model

        Returns:
            aggregated results of feature selection methods (feature ranking by weighted average)

        """

        aggregator = WeightedAveraging()
        return aggregator.aggregate(data).rank(ascending=False)


class BordaCount(AggregationBaseClass):
    """Borda count of the results of the feautre selection methods"""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Calculate the borda count of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (borda count)

        """

        data_ranks = data.rank(ascending=False)
        return data_ranks.sum(axis=1)


class BordaCountRanked(AggregationBaseClass):
    """Feature ranking by the borda count of the results of the feature selection methods."""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Calculate the feature selection rankings by the borda count of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (feature ranking by borda count)

        """

        data_ranks = data.rank(ascending=False)
        return data_ranks.sum(axis=1).rank()


class ReciprocalRanking(AggregationBaseClass):
    """Reciprocal ranking of results of feature selection methods."""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Calculate the reciprocal ranking of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (reciprocal ranking)

        """

        data_reciprocal_ranks = data.rank(ascending=False)
        data_reciprocal_ranks = data_reciprocal_ranks.applymap(lambda cell: 1 / cell)
        reciprocal_ranks = data_reciprocal_ranks.apply(
            (lambda row: 1 / np.sum(row)), axis=1
        )
        return reciprocal_ranks


class ReciprocalRankingRanked(AggregationBaseClass):
    """Feature ranking by reciprocal ranking of the results of the feature selection methods."""
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:

        """ Calculate the feature rankings by the reciprocal ranking of the results of the feature selection methods.

        Args:
            data: results of feature selection methods

        Returns:
            aggregated results of feature selection methods (feature ranking by reciprocal ranking)

        """

        aggregator = ReciprocalRanking()
        return aggregator.aggregate(data).rank()


class FeatureOccurrenceFrequency(AggregationBaseClass):
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:
        n = settings.aggregation_ensemble_feature_selection.n

        data_features_selected = data.copy()
        for column in data_features_selected:
            data_features_selected[column] = data_features_selected[column].apply(
                lambda x: 1
                if x in data_features_selected[column].nlargest(n=n).values
                else 0
            )
        votes_count = data_features_selected.sum(axis=1)
        return votes_count


class FeatureOccurrenceFrequencyRanked(AggregationBaseClass):
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:
        aggregator = FeatureOccurrenceFrequency()
        return aggregator.aggregate(data).rank(ascending=False)


class InstantRunoff(
    AggregationBaseClass
):  # always ranked! #ONLY FOR RANKED DATASETS AS INPUT
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:
        data_for_votes = data
        ranks = pd.DataFrame([None] * len(data), index=data.index)
        for i in range(len(data)):
            candidates = list(map(Candidate, data_for_votes.index))
            ballots = list(
                map(
                    Ballot,
                    [
                        list(
                            map(Candidate, data_for_votes.index[np.argsort(list(c)[1])])
                        )
                        for c in list(data_for_votes.items())
                    ],
                )
            )
            election_result = pyrankvote.instant_runoff_voting(candidates, ballots)
            winner = election_result.get_winners()[0].name
            ranks.loc[winner, :] = i + 1
            data_for_votes = data_for_votes.drop(winner)
        return ranks


class InstantRunoff2(
    AggregationBaseClass
):  # always returns ranking, ONLY USE FOR RANKED INPUT DATA
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:
        return_ranks_df = pd.DataFrame([None] * len(data), index=data.index)
        feature_mapping_dict = {
            index: feature for index, feature in enumerate(data.index)
        }
        votes_list_of_lists = list(
            [list(data.index[np.argsort(list(c)[1])]) for c in list(data.items())]
        )
        votes_mapped_list_of_lists = []

        for lst in votes_list_of_lists:
            votes_mapped_per_voter = []
            for feature_name in lst:
                for feature_key, feature in feature_mapping_dict.items():
                    if feature == feature_name:
                        votes_mapped_per_voter.append(feature_key)
            votes_mapped_list_of_lists.append(votes_mapped_per_voter)

        origin_profile = Profile(
            rankings=votes_mapped_list_of_lists, cmap=feature_mapping_dict
        )
        voting_profile = Profile(
            rankings=votes_mapped_list_of_lists, cmap=feature_mapping_dict
        )
        features_to_drop = []
        cnames_mapping_dict = {
            feature_map: feature_map
            for feature_map, feature in feature_mapping_dict.items()
        }

        for i in range(len(data)):
            winner_mapped_int = instant_runoff.choose(voting_profile)
            features_to_drop.append(cnames_mapping_dict[winner_mapped_int])
            winner = feature_mapping_dict[cnames_mapping_dict[winner_mapped_int]]
            return_ranks_df.loc[winner, :] = i + 1
            voting_profile, cnames_mapping_dict = origin_profile.remove_candidates(
                features_to_drop
            )

        return return_ranks_df


class Coombs(
    AggregationBaseClass
):  # always returns ranking, ONLY USE FOR RANKED INPUT DATA
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:
        return_ranks_df = pd.DataFrame([None] * len(data), index=data.index)
        feature_mapping_dict = {
            index: feature for index, feature in enumerate(data.index)
        }
        votes_list_of_lists = list(
            [list(data.index[np.argsort(list(c)[1])]) for c in list(data.items())]
        )
        votes_mapped_list_of_lists = []

        for lst in votes_list_of_lists:
            votes_mapped_per_voter = []
            for feature_name in lst:
                for feature_key, feature in feature_mapping_dict.items():
                    if feature == feature_name:
                        votes_mapped_per_voter.append(feature_key)
            votes_mapped_list_of_lists.append(votes_mapped_per_voter)

        origin_profile = Profile(
            rankings=votes_mapped_list_of_lists, cmap=feature_mapping_dict
        )
        voting_profile = Profile(
            rankings=votes_mapped_list_of_lists, cmap=feature_mapping_dict
        )
        features_to_drop = []
        cnames_mapping_dict = {
            feature_map: feature_map
            for feature_map, feature in feature_mapping_dict.items()
        }

        for i in range(len(data)):
            winner_mapped_int = coombs.choose(voting_profile)
            features_to_drop.append(cnames_mapping_dict[winner_mapped_int])
            winner = feature_mapping_dict[cnames_mapping_dict[winner_mapped_int]]
            return_ranks_df.loc[winner, :] = i + 1
            voting_profile, cnames_mapping_dict = origin_profile.remove_candidates(
                features_to_drop
            )

        return return_ranks_df


class Bucklin(
    AggregationBaseClass
):  # always returns ranking, ONLY USE FOR RANKED INPUT DATA
    @staticmethod
    def aggregate(data: pd.DataFrame) -> pd.DataFrame:
        return_ranks_df = pd.DataFrame([None] * len(data), index=data.index)
        feature_mapping_dict = {
            index: feature for index, feature in enumerate(data.index)
        }
        votes_list_of_lists = list(
            [list(data.index[np.argsort(list(c)[1])]) for c in list(data.items())]
        )
        votes_mapped_list_of_lists = []

        for lst in votes_list_of_lists:
            votes_mapped_per_voter = []
            for feature_name in lst:
                for feature_key, feature in feature_mapping_dict.items():
                    if feature == feature_name:
                        votes_mapped_per_voter.append(feature_key)
            votes_mapped_list_of_lists.append(votes_mapped_per_voter)

        origin_profile = Profile(
            rankings=votes_mapped_list_of_lists, cmap=feature_mapping_dict
        )
        voting_profile = Profile(
            rankings=votes_mapped_list_of_lists, cmap=feature_mapping_dict
        )
        features_to_drop = []
        cnames_mapping_dict = {
            feature_map: feature_map
            for feature_map, feature in feature_mapping_dict.items()
        }

        for i in range(len(data)):
            winner_mapped_int = bucklin.choose(voting_profile)
            features_to_drop.append(cnames_mapping_dict[winner_mapped_int])
            winner = feature_mapping_dict[cnames_mapping_dict[winner_mapped_int]]
            return_ranks_df.loc[winner, :] = i + 1
            voting_profile, cnames_mapping_dict = origin_profile.remove_candidates(
                features_to_drop
            )

        return return_ranks_df
