"""Programmatic validation of user-provided CSV data after data collection.

Checks that the data collection rules in data_guide.py were respected:
  1. sets.csv is present with the correct structure.
  2. No set_name in sets.csv is empty.
  3. No element ID is claimed by more than one set_name (ambiguous membership).
  4. No parameter CSV column mixes element IDs from multiple distinct set_name groups
     (the split-by-entity-type rule).

These checks run after UserData.load_from_csv_dir succeeds (files present, columns
typed) and before IR generation, so errors are surfaced to the user before any LLM
work is wasted.
"""

from __future__ import annotations

from collections import defaultdict

from orpilot.models.data import UserData


def validate_collected_data(user_data: UserData) -> list[str]:
    """Return a list of human-readable error strings.  Empty list = all good."""
    errors: list[str] = []

    errors.extend(_check_sets_csv(user_data))
    if errors:
        # Cannot do further checks without a valid sets.csv
        return errors

    element_to_set, ambiguity_errors = _build_element_map(user_data)
    errors.extend(ambiguity_errors)

    errors.extend(_check_mixed_entity_columns(user_data, element_to_set))

    return errors


# ---------------------------------------------------------------------------
# Internal checks
# ---------------------------------------------------------------------------

def _check_sets_csv(user_data: UserData) -> list[str]:
    """Verify sets.csv exists and has the required set_name + element columns."""
    errors: list[str] = []
    sets_table = user_data.raw_tables.get("sets")

    if sets_table is None:
        errors.append(
            "sets.csv is missing. Please provide a sets.csv file with columns "
            "'set_name' (str) and 'element' (str) listing every member of every "
            "set used in the model."
        )
        return errors

    if not sets_table:
        errors.append("sets.csv is empty — it must contain at least one row.")
        return errors

    actual_cols = set(sets_table[0].keys())
    for required in ("set_name", "element"):
        if required not in actual_cols:
            errors.append(
                f"sets.csv is missing required column '{required}'. "
                f"Found columns: {sorted(actual_cols)}. "
                f"sets.csv must have exactly two columns: 'set_name' and 'element'."
            )

    return errors


def _build_element_map(
    user_data: UserData,
) -> tuple[dict[str, str], list[str]]:
    """Build a reverse map element → set_name from sets.csv.

    Also returns ambiguity errors for element IDs that appear under more than
    one set_name — those IDs cannot be used to reliably detect mixed columns.
    Returns only the unambiguous entries in the map.
    """
    errors: list[str] = []
    element_to_sets: dict[str, set[str]] = defaultdict(set)
    set_members: dict[str, set[str]] = defaultdict(set)

    for row in user_data.raw_tables.get("sets", []):
        sn = str(row.get("set_name", "")).strip()
        el = str(row.get("element", "")).strip()
        if sn and el:
            element_to_sets[el].add(sn)
            set_members[sn].add(el)

    # Empty sets
    for sn, members in set_members.items():
        if not members:
            errors.append(
                f"Set '{sn}' in sets.csv has no elements. "
                "Every set_name must have at least one element row."
            )

    # Ambiguous element IDs
    for el, snames in element_to_sets.items():
        if len(snames) > 1:
            errors.append(
                f"Element ID '{el}' appears under multiple set names in sets.csv: "
                f"{sorted(snames)}. Element IDs must be unique across all sets to "
                "avoid ambiguous membership."
            )

    # Return only unambiguous entries
    unambiguous = {
        el: next(iter(snames))
        for el, snames in element_to_sets.items()
        if len(snames) == 1
    }
    return unambiguous, errors


def _check_mixed_entity_columns(
    user_data: UserData,
    element_to_set: dict[str, str],
) -> list[str]:
    """Detect parameter CSV columns that mix element IDs from multiple set_name groups.

    For each non-sets table, scan every column.  If a single column contains
    string values that unambiguously map to two or more different set_name groups,
    that column violates the one-index-column-per-entity-type rule.

    Only string/ID-like values are checked; float/int value columns won't contain
    entity IDs and are naturally skipped because they won't appear in element_to_set.
    """
    errors: list[str] = []

    for table_name, rows in user_data.raw_tables.items():
        if table_name == "sets" or not rows:
            continue

        for col_name in rows[0].keys():
            set_names_in_col: set[str] = set()
            for row in rows:
                v = str(row[col_name]) if row.get(col_name) is not None else ""
                if v in element_to_set:
                    set_names_in_col.add(element_to_set[v])

            if len(set_names_in_col) > 1:
                sorted_sets = sorted(set_names_in_col)
                suggested_files = " and ".join(
                    f"'{table_name}_{s}.csv'" for s in sorted_sets
                )
                errors.append(
                    f"'{table_name}.csv' column '{col_name}' mixes element IDs from "
                    f"multiple entity types: {sorted_sets}. "
                    f"Please split into separate files — one per entity type "
                    f"(e.g. {suggested_files}), each with its own unambiguous "
                    f"index column."
                )

    return errors
