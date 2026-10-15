"""Shared data splits that keep related phrases out of separate partitions."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def phrase_groups(frame: pd.DataFrame) -> list[int]:
    parent = list(range(len(frame)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def join(left, right):
        parent[find(left)] = find(right)
    # Link translations, spelling variants, conversation turns and repeated normalized text.
    for column in ("group_id", "pair_id", "conversation_id", "duplicate_key", "normalized_text"):
        if column not in frame:
            continue
        seen = {}
        for index, value in enumerate(frame[column].fillna("").astype(str)):
            value = value.strip()
            if not value:
                continue
            if value in seen:
                join(index, seen[value])
            else:
                seen[value] = index
    return [find(index) for index in range(len(frame))]


def split_data(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = phrase_groups(frame)
    strata = frame["label"].astype(str)
    if "language_pair" in frame:
        strata = strata + "|" + frame["language_pair"].astype(str)
    required = set(strata)

    if "split" in frame:
        declared = frame["split"].fillna("").astype(str).str.strip().str.lower()
        if not set(declared) <= {"train", "validation", "test"}:
            raise ValueError("Training split must be train, validation or test; demo data is for evaluation")
        group_splits = {}
        for group, partition in zip(groups, declared):
            group_splits.setdefault(group, set()).add(partition)
        if any(len(partitions) > 1 for partitions in group_splits.values()):
            raise ValueError("Related or duplicate phrases cross declared data splits")
        partitions = [frame[declared == name].copy() for name in ("train", "validation", "test")]
        if any(partition.empty for partition in partitions):
            raise ValueError("Provide non-empty train, validation and test partitions")
    elif len(set(groups)) == len(frame):
        train_indices, remainder = train_test_split(
            list(range(len(frame))), test_size=0.30, random_state=seed, stratify=strata,
        )
        val_indices, test_indices = train_test_split(
            remainder, test_size=0.50, random_state=seed, stratify=strata.iloc[remainder],
        )
        partitions = [frame.iloc[indices].copy() for indices in (train_indices, val_indices, test_indices)]
    else:
        partitions = None
        for attempt in range(200):
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed + attempt)
            train_indices, remainder = next(splitter.split(frame, groups=groups))
            if len(set(groups[index] for index in remainder)) < 2:
                continue
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + attempt)
            val_local, test_local = next(splitter.split(frame.iloc[remainder], groups=[groups[index] for index in remainder]))
            indices = (train_indices, remainder[val_local], remainder[test_local])
            if all(set(strata.iloc[part]) == required for part in indices):
                partitions = [frame.iloc[part].copy() for part in indices]
                break
        if partitions is None:
            raise ValueError("Not enough independent groups for every emotion/language in all splits. Add labelled groups or supply a safe explicit split.")
    for partition in partitions:
        present = partition["label"].astype(str)
        if "language_pair" in partition:
            present = present + "|" + partition["language_pair"].astype(str)
        if set(present) != required:
            raise ValueError("Every split must cover each supplied label/language combination")
    return tuple(partition.reset_index(drop=True) for partition in partitions)
