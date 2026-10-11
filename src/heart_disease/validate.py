"""Data validation for the training file and for app / CLI inputs.

Errors stop training; warnings are written to reports/data_validation.json.

Usage:
    python -m heart_disease.validate
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import pandas as pd

from . import config
from .data import load_raw

RANGES = {"age": (18, 100), "trestbps": (70, 250), "chol": (80, 700), "thalach": (50, 230), "oldpeak": (0, 10), "ca": (0, 3)}


@dataclass
class Check:
    name: str
    severity: str            # "error" | "warning" | "info"
    passed: bool
    failing_rows: int = 0
    detail: str = ""
    examples: list = field(default_factory=list)


@dataclass
class ValidationReport:
    rows: int
    checks: list

    @property
    def errors(self):
        return [c for c in self.checks if c.severity == "error" and not c.passed]

    @property
    def warnings(self):
        return [c for c in self.checks if c.severity == "warning" and not c.passed]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"rows": self.rows, "passed": self.ok, "errors": len(self.errors), "warnings": len(self.warnings),
                "checks": [asdict(c) for c in self.checks]}

    def summary(self) -> str:
        lines = [f"Rows checked: {self.rows}"]
        for c in self.checks:
            mark = "PASS" if c.passed else c.severity.upper()
            lines.append(f"  [{mark:<7}] {c.name}" + (f" ({c.failing_rows} rows)" if c.failing_rows else "") + (f" - {c.detail}" if c.detail else ""))
        lines.append("Result: " + ("PASSED" if self.ok else f"FAILED with {len(self.errors)} error(s)"))
        return "\n".join(lines)


class DataValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("Data validation failed:\n" + report.summary())


def check_training_data(df: pd.DataFrame) -> ValidationReport:
    checks = []
    missing_cols = [c for c in config.COLUMNS if c not in df]
    checks.append(Check("all 14 columns present", "error", not missing_cols, detail=", ".join(missing_cols)))
    if missing_cols or df.empty:
        checks.append(Check("file is not empty", "error", not df.empty))
        return ValidationReport(len(df), checks)
    bad = df["num"].isna() | ~df["num"].isin([0, 1, 2, 3, 4])
    checks.append(Check("diagnosis (num) present and 0-4", "error", not bad.any(), int(bad.sum())))
    dups = int(df.duplicated().sum())
    checks.append(Check("no duplicate patients", "warning", dups == 0, dups))
    for c in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
        n = int(df[c].isna().sum())
        checks.append(Check(f"{c} has no missing values", "warning", n == 0, n, detail="imputed in the pipeline" if n else ""))
    for c, (lo, hi) in RANGES.items():
        m = df[c].notna() & ((df[c] < lo) | (df[c] > hi))
        checks.append(Check(f"{c} within {lo}-{hi}", "warning", not m.any(), int(m.sum()),
                            examples=df.loc[m, [c]].head(3).to_dict("records")))
    for c, codes in config.CODES.items():
        m = df[c].notna() & ~df[c].isin(list(codes))
        checks.append(Check(f"{c} uses known codes {sorted(codes)}", "warning", not m.any(), int(m.sum())))
    share = float((df["num"] > 0).mean())
    checks.append(Check("classes are balanced enough (30-70% disease)", "warning", 0.3 <= share <= 0.7, detail=f"{share:.0%} with disease"))
    return ValidationReport(len(df), checks)


def assert_valid(df: pd.DataFrame) -> ValidationReport:
    report = check_training_data(df)
    if not report.ok:
        raise DataValidationError(report)
    return report


def check_inputs(row: dict) -> list[str]:
    """Plain-language warnings for one patient entered in the app or CLI. Never blocks a prediction."""
    msgs = []
    for c, (lo, hi) in RANGES.items():
        v = row.get(c)
        if v is not None and not pd.isna(v) and not (lo <= float(v) <= hi):
            msgs.append(f"{config.LABELS[c]} = {v} is outside the range seen in training ({lo}-{hi}).")
    for c, codes in config.CODES.items():
        v = row.get(c)
        if v is not None and not pd.isna(v) and int(v) not in codes:
            msgs.append(f"{config.LABELS[c]} = {v} is not a known code ({', '.join(map(str, codes))}).")
    missing = [config.LABELS[c] for c in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES if row.get(c) is None or pd.isna(row.get(c))]
    if missing:
        msgs.append("Missing and imputed: " + ", ".join(missing) + ".")
    return msgs


def main() -> None:
    report = check_training_data(load_raw())
    print(report.summary())
    config.REPORTS_DIR.mkdir(exist_ok=True)
    with open(config.REPORTS_DIR / "data_validation.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
