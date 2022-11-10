# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import dataclasses
import enum
import itertools
from collections import defaultdict
from math import sqrt
from typing import Collection, Dict, List, NamedTuple, Optional, TextIO, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score


class Dataset(enum.Enum):
    QUERIES = "Q"
    REFS = "R"


def format_video_id(video_id: Union[str, int], dataset: Optional[Dataset]) -> str:
    if isinstance(video_id, (int, np.integer)):
        if dataset is None:
            raise ValueError(
                "Unable to convert integer video_id without a Dataset enum"
            )
        return f"{dataset.value}{video_id:06d}"
    assert isinstance(
        video_id, str
    ), f"unexpected video_id: {video_id} of type {type(video_id)}"
    if dataset is not None:
        assert (
            video_id[0] == dataset.value
        ), f"dataset mismatch? got {video_id} for dataset {dataset}"
    return video_id


@dataclasses.dataclass
class CandidatePair:
    query_id: str
    ref_id: str
    score: float

    @classmethod
    def to_dataframe(
        cls,
        candidates: Collection["CandidatePair"],
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "query_id": format_video_id(c.query_id, Dataset.QUERIES),
                    "ref_id": format_video_id(c.ref_id, Dataset.REFS),
                    "score": c.score,
                }
                for c in candidates
            ],
        )

    @classmethod
    def write_csv(
        cls, candidates: Collection["CandidatePair"], file: Union[str, TextIO]
    ):
        df = cls.to_dataframe(candidates)
        df.to_csv(file, index=False)

    @classmethod
    def read_csv(cls, file: Union[str, TextIO]) -> List["CandidatePair"]:
        df = pd.read_csv(file)
        pairs = []
        for _, row in df.iterrows():
            query_id = format_video_id(row.query_id, Dataset.QUERIES)
            ref_id = format_video_id(row.ref_id, Dataset.REFS)
            pairs.append(
                CandidatePair(query_id=query_id, ref_id=ref_id, score=row.score)
            )
        return pairs

    @classmethod
    def from_matches(cls, matches: Collection["Match"]) -> List["CandidatePair"]:
        scores = collections.defaultdict(float)
        for match in matches:
            key = (match.query_id, match.ref_id)
            scores[key] = max(match.score, scores[key])
        return [
            CandidatePair(query_id=query_id, ref_id=ref_id, score=score)
            for ((query_id, ref_id), score) in scores.items()
        ]


@dataclasses.dataclass
class PrecisionRecallCurve:
    precisions: np.ndarray
    recalls: np.ndarray
    scores: np.ndarray

    def plot(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
            ax.set_xlabel("recall")
            ax.set_ylabel("precision")
            ax.set_xlim(0, 1.05)
            ax.set_ylim(0, 1.05)
        ax.plot(self.recalls, self.precisions, **kwargs)
        return ax


@dataclasses.dataclass
class AveragePrecision:
    ap: float
    pr_curve: PrecisionRecallCurve
    simple_ap: Optional[float] = None


class Intervals:

    # Non-overlapping, ordered by interval start.
    intervals: List[Tuple[float, float]]

    def __init__(self, intervals: Optional[List[Tuple[float, float]]] = None):
        self.intervals = intervals or []
        self._dedup()

    def add(self, interval: Tuple[float, float]):
        """Add an interval."""
        self.intervals.append(interval)
        self._dedup()

    def union(self, intervals: "Intervals") -> "Intervals":
        return Intervals(self.intervals + intervals.intervals)

    def total_length(self) -> float:
        length = 0.0
        for start, end in self.intervals:
            length += end - start
        return length

    def intersect_length(self, intervals: "Intervals") -> float:
        """Compute the total_length of the intersection of two Intervals.

        This works by taking the sum of their lengths, and subtracting
        the length of their union.

        |A n B| = |A| + |B| - |A U B|
        """
        union = self.union(intervals)
        return self.total_length() + intervals.total_length() - union.total_length()

    def _dedup(self):
        if len(self.intervals) <= 1:
            return
        deduped = []
        intervals = sorted(self.intervals)
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end:
                # Overlap case
                current_end = max(end, current_end)
            else:
                # Non-overlap case
                deduped.append((current_start, current_end))
                current_start, current_end = start, end
        deduped.append((current_start, current_end))
        self.intervals = deduped

    def __str__(self):
        return str(self.intervals)

    __repr__ = __str__


class Axis(enum.Enum):
    QUERY = enum.auto()
    REF = enum.auto()


class Match(NamedTuple):
    """A ground-truth match or predicted match."""

    query_id: str
    ref_id: str
    score: float
    query_start: float
    query_end: float
    ref_start: float
    ref_end: float

    def pair_id(self):
        return (self.query_id, self.ref_id)

    def interval(self, axis: Axis) -> Tuple[float, float]:
        if axis == Axis.QUERY:
            return (self.query_start, self.query_end)
        else:
            return (self.ref_start, self.ref_end)

    def intersection_area(self, bbox: "Match") -> float:
        # Compute the intersection boarders
        inter_q_start = max(self.query_start, bbox.query_start)
        inter_r_start = max(self.ref_start, bbox.ref_start)
        inter_q_end = min(self.query_end, bbox.query_end)
        inter_r_end = min(self.ref_end, bbox.ref_end)

        # Compute the area of intersection rectangle
        return abs(
            max((inter_q_end - inter_q_start, 0))
            * max((inter_r_end - inter_r_start), 0)
        )

    def overlaps(self, bbox: "Match") -> bool:
        return self.intersection_area(bbox) > 0.0

    @classmethod
    def write_csv(cls, matches: Collection["Match"], file: Union[str, TextIO]):
        df = pd.DataFrame([match._asdict() for match in matches], columns=cls._fields)
        df.to_csv(file, index=False)

    @classmethod
    def read_csv(
        cls, file: Union[str, TextIO], is_gt=False, check=True
    ) -> List["Match"]:
        df = pd.read_csv(file)
        df["query_id"] = df.query_id.map(lambda x: format_video_id(x, Dataset.QUERIES))
        df["ref_id"] = df.ref_id.map(lambda x: format_video_id(x, Dataset.REFS))
        if is_gt:
            df["score"] = 1.0
        if check:
            for field in cls._fields:
                assert not df[field].isna().any()
        return [Match(**record) for record in df.to_dict("records")]


class VideoPair:
    """A video pair item that contains information regarding the gt and pred bboxes.

    Provide functionalities for the combination of new predictions with the
    existing ones and the computation of their intersection with the gt bboxes,
    ignoring the gt bboxes that do not overlap with any prediction.
    """

    gts: List[Match]
    preds: List[Match]

    def __init__(
        self,
    ):
        self.intersections = {axis: 0.0 for axis in Axis}
        self.totals = {axis: 0.0 for axis in Axis}
        self.gts = []
        self.preds = []

    def total_gt_length(self, axis: Axis) -> float:
        return Intervals([gt.interval(axis) for gt in self.gts]).total_length()

    def total_pred_length(self, axis: Axis) -> float:
        return Intervals([pred.interval(axis) for pred in self.preds]).total_length()

    def gt_overlaps(self, gt: Match) -> bool:
        """Checks if the provided gt bbox overlaps with at least one pred bbox."""
        for pred in self.preds:
            if gt.overlaps(pred):
                return True
        return False

    def add_gt(self, bbox: Match):
        self.gts.append(bbox)

    def add_prediction(
        self, bbox: Match
    ) -> Tuple[Dict[Axis, float], Dict[Axis, float]]:
        """Add a prediction to the corresponding list and calculates the
        differences in the intersections with the gt and the total video
        length covered for both query and reference axes.
        """
        self.preds.append(bbox)

        # A subset of GTs to consider for intersection (but not total GT length).
        gts_to_consider = [gt for gt in self.gts if self.gt_overlaps(gt)]

        intersect_deltas = {}
        total_deltas = {}

        for axis in Axis:
            pred_ints = Intervals([pred.interval(axis) for pred in self.preds])
            gt_ints = Intervals([gt.interval(axis) for gt in gts_to_consider])
            # New intersection and total length on this axis
            intersect_length = pred_ints.intersect_length(gt_ints)
            prediction_length = pred_ints.total_length()
            # Compute differences
            intersect_deltas[axis] = intersect_length - self.intersections[axis]
            total_deltas[axis] = prediction_length - self.totals[axis]
            # Update with new values
            self.intersections[axis] = intersect_length
            self.totals[axis] = prediction_length

        return intersect_deltas, total_deltas


def match_metric(
    gts: Collection[Match],
    predictions: Collection[Match],
) -> AveragePrecision:
    """Matching track metric:

    Computes the AP based on the VCSL approach for the
    calculation of Precision and Recall.

    AP = \sum_{i=1}^N P(i) ΔR(i)

    where, P(i) = sqrt(P_q * P_r) and R(i) = sqrt(R_q * R_r)
    calculated as in the VCSL.
    """  # noqa: W605

    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)

    # Initialize video pairs and load their gt bboxs
    video_pairs = defaultdict(VideoPair)
    for gt in gts:
        video_pairs[gt.pair_id()].add_gt(gt)

    # Get the total gt length for each axis
    gt_total_lengths = {axis: 0.0 for axis in Axis}
    for _, v in video_pairs.items():
        for axis in Axis:
            gt_total_lengths[axis] += v.total_gt_length(axis)

    # Loop through the predictions
    recall = 0.0
    metric = 0.0
    intersections = {axis: 0.0 for axis in Axis}
    totals = {axis: 0.0 for axis in Axis}
    pr_recalls = []
    pr_precisions = []
    pr_scores = []

    # Update metrics for all predictions with the same score as a group.
    for score, prediction_group in itertools.groupby(
        predictions, key=lambda x: x.score
    ):
        for prediction in prediction_group:
            # Given new predictions, we only need the differences in the intersection with
            # gt and total video length covered for both query and reference axes.
            # This derives from the sum of differences for every pair id
            intersection_deltas, total_deltas = video_pairs[
                prediction.pair_id()
            ].add_prediction(prediction)
            for axis in Axis:
                # Accumulate the differences to the corresponding values
                intersections[axis] += intersection_deltas[axis]
                totals[axis] += total_deltas[axis]

        recalls = {}
        precisions = {}
        for axis in Axis:
            recalls[axis] = intersections[axis] / gt_total_lengths[axis]
            precisions[axis] = intersections[axis] / totals[axis]

        new_recall = sqrt(recalls[Axis.QUERY] * recalls[Axis.REF])
        precision = sqrt(precisions[Axis.QUERY] * precisions[Axis.REF])

        # Compute metric
        delta_recall = new_recall - recall
        metric += precision * delta_recall
        recall = new_recall
        if delta_recall > 0:
            pr_recalls.append(recall)
            pr_precisions.append(precision)
            pr_scores.append(score)

    curve = PrecisionRecallCurve(
        np.array(pr_precisions), np.array(pr_recalls), np.array(pr_scores)
    )
    return AveragePrecision(metric, curve)


@dataclasses.dataclass
class MatchingTrackMetrics:
    # Our main evaluation metric.
    segment_ap: AveragePrecision
    # This metric reflects only pairwise matching, and not localization.
    pairwise_micro_ap: AveragePrecision


def evaluate_matching_track(
    ground_truth_filename: str, predictions_filename: str
) -> MatchingTrackMetrics:
    """Matching track evaluation.

    Predictions are expected to be a CSV file, with a column names in the header.
    The following columns must be present, in any order:
        query_id: str, the ID of the query for this match
        ref_id: str, the ID of the reference for this match
        query_start: float, the start of the query segment in seconds
        query_end: float, the end of the query segment in seconds
        ref_start: float, the start of the reference segment in seconds
        ref_end: float, the end of the reference segment in seconds
        score: float, the score of this prediction (a higher score indicates a
            more confident prediction)

    Note that ground-truth matches are specified using the same format, but score
    is not used.
    """
    gt = Match.read_csv(ground_truth_filename, is_gt=True)
    predictions = Match.read_csv(predictions_filename)
    metric = match_metric(gt, predictions)
    # Auxiliary metric: pairwise uAP
    gt_pairs = CandidatePair.from_matches(gt)
    pairs = CandidatePair.from_matches(predictions)
    pair_ap = average_precision(gt_pairs, pairs)
    return MatchingTrackMetrics(segment_ap=metric, pairwise_micro_ap=pair_ap)


def average_precision(
    ground_truth: Collection[CandidatePair], predictions: Collection[CandidatePair]
) -> AveragePrecision:
    gt_pairs = {(pair.query_id, pair.ref_id) for pair in ground_truth}
    if len(gt_pairs) != len(ground_truth):
        raise AssertionError("Duplicates detected in ground truth")
    predicted_pairs = {(pair.query_id, pair.ref_id) for pair in predictions}
    if len(predicted_pairs) != len(predictions):
        raise AssertionError("Duplicates detected in predictions")

    # AP calculation that aligns with DrivenData's backend implementation.
    canonical_ap = drivendata_average_precision(
        predicted=CandidatePair.to_dataframe(predictions),
        ground_truth=CandidatePair.to_dataframe(ground_truth),
    )

    predictions = sorted(predictions, key=lambda x: x.score, reverse=True)
    scores = np.array([pair.score for pair in predictions])
    correct = np.array(
        [(pair.query_id, pair.ref_id) in gt_pairs for pair in predictions]
    )
    total_pairs = len(gt_pairs)
    # precision = correct_so_far / total_pairs_so_far
    cumulative_correct = np.cumsum(correct)
    cumulative_predicted = np.arange(len(correct)) + 1
    recall = cumulative_correct / total_pairs
    precision = cumulative_correct / cumulative_predicted
    # Simple AP computation.
    simple_ap = np.sum(precision * correct) / total_pairs
    # Get precision and recall where correct is true
    indices = np.nonzero(correct)[0]
    curve = PrecisionRecallCurve(precision[indices], recall[indices], scores[indices])
    return AveragePrecision(ap=canonical_ap, pr_curve=curve, simple_ap=simple_ap)


def drivendata_average_precision(
    predicted: pd.DataFrame,
    ground_truth: pd.DataFrame,
):
    """Canonical AP implementation used for the challenge."""
    SCORE_COL = "score"
    QUERY_ID_COL = "query_id"
    DATABASE_ID_COL = "ref_id"
    actual = ground_truth[["query_id", "ref_id"]]
    if (
        not np.isfinite(predicted[SCORE_COL]).all()
        or np.isnan(predicted[SCORE_COL]).any()
    ):
        raise ValueError("Scores must be finite.")

    predicted = predicted.sort_values(SCORE_COL, ascending=False)

    merged = predicted.merge(
        right=actual.assign(actual=1.0),
        how="left",
        on=[QUERY_ID_COL, DATABASE_ID_COL],
    ).fillna({"actual": 0.0})

    # We may not predict for every ground truth, so calculate unadjusted AP then adjust it
    unadjusted_ap = (
        average_precision_score(merged["actual"].values, merged[SCORE_COL].values)
        if merged["actual"].sum()
        else 0.0
    )
    # Rescale average precisions based on total ground truth positive counts
    predicted_n_pos = int(merged["actual"].sum())

    # avoid rows added to validate query ids, will have blank ref_id
    actual_n_pos = int(actual[DATABASE_ID_COL].notna().sum())

    adjusted_ap = unadjusted_ap * (predicted_n_pos / actual_n_pos)
    return adjusted_ap
