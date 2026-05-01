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
    optional: bool = Field(
        default=False,
        description="If True, the file may be absent; missing it is not an error.",
    )


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

        Raises ``FileNotFoundError`` listing every missing file.
        Raises ``ValueError`` listing every missing column or bad cell value.
        """
        dir_path = Path(directory)

        # Validate all required files exist (optional files may be absent)
        missing = [
            spec.filename
            for spec in specs
            if not spec.optional and not (dir_path / spec.filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing CSV file(s) in {directory}: {', '.join(missing)}"
            )

        validation_errors: list[str] = []
        raw_tables: dict[str, list[dict[str, Any]]] = {}
        for spec in specs:
            filepath = dir_path / spec.filename
            if spec.optional and not filepath.is_file():
                continue
            col_dtypes = {c.name: c.dtype for c in spec.columns}
            with open(filepath, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                actual_columns = set(reader.fieldnames or [])

                # Check that every declared column is present in the CSV header
                expected_columns = {c.name for c in spec.columns}
                missing_cols = expected_columns - actual_columns
                if missing_cols:
                    validation_errors.append(
                        f"{spec.filename}: missing column(s): {', '.join(sorted(missing_cols))}"
                    )

                rows: list[dict[str, Any]] = []
                for row_num, row in enumerate(reader, start=2):  # row 1 is the header
                    typed_row: dict[str, Any] = {}
                    for key, value in row.items():
                        dtype = col_dtypes.get(key, "str")
                        try:
                            typed_row[key] = _cast_value(value, dtype)
                        except ValueError as exc:
                            validation_errors.append(
                                f"{spec.filename} row {row_num}, column '{key}': {exc}"
                            )
                            typed_row[key] = value  # keep original so other errors can still be found
                    rows.append(typed_row)

                # Use filename stem as table name
                table_name = Path(spec.filename).stem
                raw_tables[table_name] = rows

        if validation_errors:
            raise ValueError(
                "CSV data validation failed:\n"
                + "\n".join(f"  - {e}" for e in validation_errors)
            )

        # Normalize set-member values to str across all parameter tables.
        # sets.csv always stores element IDs as str (element column dtype is str).
        # Parameter tables may have index columns typed as int (e.g. period_id: int),
        # which after dtype casting produces int keys. Lookups then silently miss
        # because the loop variable 't' from sets.csv is always str '1', not int 1.
        # Converting any non-str value whose str() form matches a set element ID
        # ensures all set-member references are consistently str throughout data.
        all_elements: set[str] = {
            str(r["element"])
            for r in raw_tables.get("sets", [])
            if r.get("element") is not None
        }
        if all_elements:
            for tname, rows in raw_tables.items():
                if tname == "sets":
                    continue
                for row in rows:
                    for key in list(row):
                        val = row[key]
                        if not isinstance(val, str) and str(val) in all_elements:
                            row[key] = str(val)

        return cls(
            raw_tables=raw_tables,
            csv_specs=specs,
            csv_dir=directory,
        )


def _cast_value(value: str, dtype: str) -> Any:
    """Cast a string *value* to *dtype*.

    Raises ``ValueError`` with a descriptive message if the cast fails.
    """
    dtype = dtype.lower().strip()
    if dtype in ("int", "integer"):
        try:
            return int(value)
        except (ValueError, TypeError):
            raise ValueError(f"expected int, got '{value}'")
    if dtype in ("float", "double", "number"):
        try:
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"expected float, got '{value}'")
    if dtype in ("bool", "boolean"):
        return value.lower() in ("true", "1", "yes")
    return value
