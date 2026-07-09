"""Typed result rows and JSON/CSV persistence."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field


@dataclass
class FidelityRow:
    weight: float
    per_attribute: dict[str, float]
    mean: float

    @classmethod
    def from_mapping(cls, row: dict, attrs: list[str]):
        if "per_attribute" in row:
            per_attribute = {attr: float(row["per_attribute"][attr]) for attr in attrs}
        else:
            per_attribute = {attr: float(row[attr]) for attr in attrs}
        return cls(
            weight=float(row["weight"]),
            per_attribute=per_attribute,
            mean=float(row["mean"]),
        )

    def to_flat_dict(self) -> dict:
        return {"weight": self.weight, **self.per_attribute, "mean": self.mean}


@dataclass
class DiversityRow:
    weight: float
    diversity: float

    def to_flat_dict(self) -> dict:
        return {"weight": self.weight, "diversity": self.diversity}


@dataclass
class VarianceRow:
    weight: float
    mean_fidelity: float
    std_fidelity: float
    runs: list[float] = field(default_factory=list)

    def to_flat_dict(self) -> dict:
        return {
            "weight": self.weight,
            "mean_fidelity": self.mean_fidelity,
            "std_fidelity": self.std_fidelity,
            "runs": list(self.runs),
        }


def normalize_fidelity_rows(rows: list[dict], attrs: list[str]) -> list[FidelityRow]:
    return [FidelityRow.from_mapping(row, attrs) for row in rows]


def load_result_file(path):
    with open(path) as f:
        obj = json.load(f)
    meta = obj["meta"]
    rows = obj["results"]
    if meta.get("test", "fidelity") == "fidelity":
        rows = [row.to_flat_dict() for row in normalize_fidelity_rows(rows, meta["attrs"])]
    return meta, sorted(rows, key=lambda r: r["weight"])


def fieldnames_for(test: str, attrs: list[str]) -> list[str]:
    if test == "fidelity":
        return ["weight"] + list(attrs) + ["mean"]
    if test == "diversity":
        return ["weight", "diversity"]
    if test == "variance":
        return ["weight", "mean_fidelity", "std_fidelity"]
    raise ValueError(f"unknown result test {test!r}")


def save_results(results_dir, tag, test, meta, rows, fieldnames=None):
    os.makedirs(results_dir, exist_ok=True)
    meta = {"schema_version": 2, **meta}
    flat_rows = [row.to_flat_dict() if hasattr(row, "to_flat_dict") else row for row in rows]
    fieldnames = fieldnames or fieldnames_for(test, meta.get("attrs", []))

    json_path = os.path.join(results_dir, f"{tag}_{test}.json")
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": flat_rows}, f, indent=2)

    csv_path = os.path.join(results_dir, f"{tag}_{test}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    return json_path, csv_path
