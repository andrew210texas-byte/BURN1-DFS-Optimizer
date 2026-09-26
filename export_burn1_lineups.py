from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SLATE = "2026_week3_sunday_main"

SLOTS = (
    "QB",
    "RB1",
    "RB2",
    "WR1",
    "WR2",
    "WR3",
    "TE",
    "FLEX",
    "DST",
)

SITES = {
    "dk": {
        "name": "DraftKings",
        "slug": "draftkings",
        "wide_file": ROOT / "modeling" / "output" / "burn1" / "burn1_draftkings_2026_week3_sunday_main_lineups_wide.csv",
        "salary_file": ROOT / "data" / "salaries" / "dk_2026_week3_sunday_main.csv",
        "id_column": "ID",
        "display_column": "Name + ID",
    },
    "fd": {
        "name": "FanDuel",
        "slug": "fanduel",
        "wide_file": ROOT / "modeling" / "output" / "burn1" / "burn1_fanduel_2026_week3_sunday_main_lineups_wide.csv",
        "salary_file": ROOT / "data" / "salaries" / "fd_2026_week3_sunday_main.csv",
        "id_column": "Id",
        "display_column": "Nickname",
    },
}


def clean(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_salary_lookup(site_key: str, salary_file: Path):
    site = SITES[site_key]

    if not salary_file.exists():
        raise FileNotFoundError(f"Salary file not found: {salary_file}")

    salary = pd.read_csv(
        salary_file,
        dtype=str,
        keep_default_na=False,
    )

    required = {
        site["id_column"],
        site["display_column"],
    }

    missing = required - set(salary.columns)

    if missing:
        raise ValueError(
            "Salary file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    lookup = {}

    for _, row in salary.iterrows():
        player_id = clean(row[site["id_column"]])

        if not player_id:
            continue

        if site_key == "dk":
            display_value = clean(row[site["display_column"]])
        else:
            display_value = (
                f"{clean(row[site['display_column']])} ({player_id})"
            )

        lookup[player_id] = display_value

    return lookup


def validate_wide(wide: pd.DataFrame, salary_lookup: dict[str, str]):
    required = {
        "lineup_number",
        "total_salary",
        "total_projection",
    }

    required.update(
        f"{slot}_id"
        for slot in SLOTS
    )

    missing = required - set(wide.columns)

    if missing:
        raise ValueError(
            "BURN1 wide lineup file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if wide.empty:
        raise ValueError("BURN1 wide lineup file contains no lineups.")

    unknown_ids = set()

    for slot in SLOTS:
        column = f"{slot}_id"

        for value in wide[column]:
            player_id = clean(value)

            if not player_id:
                raise ValueError(f"Blank player ID found in {column}.")

            if player_id not in salary_lookup:
                unknown_ids.add(player_id)

    if unknown_ids:
        preview = ", ".join(sorted(unknown_ids)[:20])

        raise ValueError(
            "BURN1 lineup IDs were not found in the site salary file: "
            + preview
        )


def build_exports(wide: pd.DataFrame, salary_lookup: dict[str, str]):
    id_rows = []
    review_rows = []

    for _, lineup in wide.iterrows():
        id_row = {
            "lineup_number": int(lineup["lineup_number"]),
        }

        review_row = {
            "lineup_number": int(lineup["lineup_number"]),
        }

        for slot in SLOTS:
            player_id = clean(lineup[f"{slot}_id"])

            id_row[slot] = player_id
            review_row[slot] = salary_lookup[player_id]

        id_row["total_salary"] = int(float(lineup["total_salary"]))
        id_row["total_projection"] = float(lineup["total_projection"])

        review_row["total_salary"] = int(float(lineup["total_salary"]))
        review_row["total_projection"] = float(lineup["total_projection"])

        id_rows.append(id_row)
        review_rows.append(review_row)

    return (
        pd.DataFrame(id_rows),
        pd.DataFrame(review_rows),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export completed BURN1 NFL portfolios into site-identity roster CSVs."
    )

    parser.add_argument(
        "--site",
        choices=("dk", "fd"),
        required=True,
    )

    parser.add_argument(
        "--slate",
        default=DEFAULT_SLATE,
    )

    parser.add_argument(
        "--wide-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--salary-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "modeling" / "output" / "burn1",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    site = SITES[args.site]

    wide_file = args.wide_file or site["wide_file"]
    salary_file = args.salary_file or site["salary_file"]

    if not wide_file.exists():
        raise FileNotFoundError(
            f"BURN1 wide lineup file not found: {wide_file}"
        )

    wide = pd.read_csv(
        wide_file,
        dtype=str,
        keep_default_na=False,
    )

    salary_lookup = load_salary_lookup(
        args.site,
        salary_file,
    )

    validate_wide(
        wide,
        salary_lookup,
    )

    ids_export, review_export = build_exports(
        wide,
        salary_lookup,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = f"burn1_{site['slug']}_{args.slate}"

    ids_file = args.output_dir / f"{prefix}_site_ids.csv"
    review_file = args.output_dir / f"{prefix}_site_review.csv"

    ids_export.to_csv(
        ids_file,
        index=False,
    )

    review_export.to_csv(
        review_file,
        index=False,
    )

    print()
    print("=" * 92)
    print(f"BURN1 {site['name'].upper()} SITE EXPORT COMPLETE")
    print("=" * 92)
    print(f"Lineups exported:           {len(ids_export)}")
    print("Salary identity validation: PASSED")
    print()
    print(f"ID export:     {ids_file}")
    print(f"Review export: {review_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
