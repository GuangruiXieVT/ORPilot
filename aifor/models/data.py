"""User-provided data schemas."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DataParameter(BaseModel):
    """A single named data parameter with its value."""

    name: str
    description: str = ""
    value: Any = None


class CsvColumnSpec(BaseModel):
    """Schema for a single column in a CSV file."""

    name: str
    dtype: str = Field(description="Expected data type (e.g. 'int', 'float', 'str')")
    description: str = ""


class CsvFileSpec(BaseModel):
    """Specification for a CSV file that the user must provide."""

    filename: str
    description: str = ""
    columns: list[CsvColumnSpec] = Field(default_factory=list)


class UserData(BaseModel):
    """Container for all user-provided data for an OR problem."""

    parameters: list[DataParameter] = Field(default_factory=list)
    raw_tables: dict[str, list[dict[str, Any]]] = Field(
        default_factory=dict,
        description="Named tabular data (e.g. CSV rows) provided by user",
    )
    raw_text: str = Field("", description="Any raw text/notes the user provided about data")
    csv_specs: list[CsvFileSpec] = Field(default_factory=list)
    csv_dir: str = Field("", description="Directory where CSV files are stored")

    def as_dict(self) -> dict[str, Any]:
        """Flatten parameters into a simple dict for solver code."""
        result: dict[str, Any] = {}
        for p in self.parameters:
            result[p.name] = p.value
        for name, rows in self.raw_tables.items():
            result[name] = rows
        return result

    @classmethod
    def load_from_csv_dir(
        cls,
        directory: str,
        specs: list[CsvFileSpec],
    ) -> UserData:
        """Load CSV files from *directory* according to *specs*.

        Raises ``FileNotFoundError`` with a clear message listing every
        missing file.
        """
        dir_path = Path(directory)

        # Validate all expected files exist
        missing = [
            spec.filename
            for spec in specs
            if not (dir_path / spec.filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing CSV file(s) in {directory}: {', '.join(missing)}"
            )

        raw_tables: dict[str, list[dict[str, Any]]] = {}
        for spec in specs:
            filepath = dir_path / spec.filename
            with open(filepath, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows: list[dict[str, Any]] = []
                for row in reader:
                    # Cast values based on column dtype specs
                    typed_row: dict[str, Any] = {}
                    col_dtypes = {c.name: c.dtype for c in spec.columns}
                    for key, value in row.items():
                        dtype = col_dtypes.get(key, "str")
                        typed_row[key] = _cast_value(value, dtype)
                    rows.append(typed_row)
                # Use filename stem as table name
                table_name = Path(spec.filename).stem
                raw_tables[table_name] = rows

        return cls(
            raw_tables=raw_tables,
            csv_specs=specs,
            csv_dir=directory,
        )


def _cast_value(value: str, dtype: str) -> Any:
    """Best-effort cast of a string *value* to *dtype*."""
    dtype = dtype.lower().strip()
    if dtype in ("int", "integer"):
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if dtype in ("float", "double", "number"):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if dtype in ("bool", "boolean"):
        return value.lower() in ("true", "1", "yes")
    return value
