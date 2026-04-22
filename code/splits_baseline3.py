"""Split helpers for Baseline 3."""

from __future__ import annotations

from binding_graph_preprocessing.constants import DEFAULT_VALID_SAMPLE_IDS

SAMPLE_ORDER = list(DEFAULT_VALID_SAMPLE_IDS)


def resolve_baseline3_split(
    split_mode: str,
    split_round: int | None = None,
    test_sample_id: str | None = None,
    val_mode: str = "deterministic",
    val_sample_id: str | None = None,
) -> dict[str, object]:
    if split_mode == "rotating_train_val_test":
        if split_round is None:
            raise ValueError("split_round is required for rotating_train_val_test")
        if split_round < 1 or split_round > len(SAMPLE_ORDER):
            raise ValueError(f"split_round must be in [1, {len(SAMPLE_ORDER)}], got {split_round}")

        test_index = split_round - 1
        val_index = (test_index + 1) % len(SAMPLE_ORDER)
        train_ids = [
            sample_id
            for idx, sample_id in enumerate(SAMPLE_ORDER)
            if idx not in {test_index, val_index}
        ]
        return {
            "split_mode": split_mode,
            "split_round": split_round,
            "train_sample_ids": train_ids,
            "val_sample_ids": [SAMPLE_ORDER[val_index]],
            "test_sample_ids": [SAMPLE_ORDER[test_index]],
        }

    if split_mode == "leave_one_out":
        if test_sample_id is None:
            raise ValueError("test_sample_id is required for leave_one_out")
        if test_sample_id not in SAMPLE_ORDER:
            raise ValueError(f"Unknown test_sample_id: {test_sample_id}")

        candidate_ids = [sample_id for sample_id in SAMPLE_ORDER if sample_id != test_sample_id]
        if val_mode == "none":
            return {
                "split_mode": split_mode,
                "split_round": None,
                "train_sample_ids": candidate_ids,
                "val_sample_ids": [],
                "test_sample_ids": [test_sample_id],
            }

        if val_mode == "explicit":
            if val_sample_id is None:
                raise ValueError("val_sample_id is required when val_mode='explicit'")
            if val_sample_id == test_sample_id or val_sample_id not in candidate_ids:
                raise ValueError("val_sample_id must be a non-test valid sample")
            val_ids = [val_sample_id]
        elif val_mode == "deterministic":
            val_ids = [candidate_ids[0]]
        else:
            raise ValueError(f"Unsupported val_mode: {val_mode}")

        train_ids = [sample_id for sample_id in candidate_ids if sample_id not in val_ids]
        return {
            "split_mode": split_mode,
            "split_round": None,
            "train_sample_ids": train_ids,
            "val_sample_ids": val_ids,
            "test_sample_ids": [test_sample_id],
        }

    raise ValueError(f"Unsupported split_mode: {split_mode}")

