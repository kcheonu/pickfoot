"""Command-line entry point for training per-league local models."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from pickfoot.backtest import expanding_window_backtest
from pickfoot.data import CSVValidationError, filter_league, load_matches
from pickfoot.model import ModelError, save_artifact, train_model
from pickfoot.service import merged_matches
from pickfoot.storage import StorageError, read_uploaded_matches, save_backtest_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pickfoot", description="Train local PickFoot models.")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train independent per-league PRE_MATCH models.")
    train.add_argument("--csv", default="data/raw/matches.csv", help="Authorized match-results CSV.")
    train.add_argument("--artifact-dir", default="models", help="Directory for local model files.")
    train.add_argument("--source-name", required=True, help="Dataset provider/name for provenance.")
    train.add_argument(
        "--license-reference",
        required=True,
        help="License/terms reference or documented private permission record.",
    )
    train.add_argument(
        "--source-collected-at",
        help="Optional known UTC ISO-8601 time when you obtained this local dataset.",
    )
    train.add_argument("--source-url", help="Public source URL or documented local source reference.")
    train.add_argument(
        "--confirm-usage-rights",
        action="store_true",
        help="Confirm you verified that the data permits this local ML training use.",
    )
    train.add_argument(
        "--synthetic-test-data",
        action="store_true",
        help="Mark this artifact as synthetic test-only data; never treat it as a real forecast.",
    )
    backtest = commands.add_parser(
        "backtest",
        help="Run a time-safe expanding-window historical PRE_MATCH reconstruction.",
    )
    backtest.add_argument("--csv", default="data/raw/matches.csv", help="Authorized match-results CSV.")
    backtest.add_argument(
        "--database",
        default="data/runtime/pickfoot.sqlite3",
        help="Local upload/prediction database; uploaded results are included when present.",
    )
    backtest.add_argument(
        "--output",
        default="data/runtime/walk_forward_backtest.json",
        help="Path for the generated backtest report.",
    )
    backtest.add_argument(
        "--confirm-usage-rights",
        action="store_true",
        help="Confirm that this dataset permits local model evaluation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "backtest":
        if not args.confirm_usage_rights:
            print(
                "Backtest stopped: verify that this dataset permits local model evaluation, "
                "then pass --confirm-usage-rights.",
                file=sys.stderr,
            )
            return 2
        try:
            base_matches = load_matches(args.csv)
            matches = merged_matches(base_matches, read_uploaded_matches(args.database))
            report = expanding_window_backtest(matches)
            save_backtest_results(args.output, report)
        except (CSVValidationError, ModelError, StorageError) as exc:
            print(f"Backtest failed: {exc}", file=sys.stderr)
            return 2
        summary = {
            "report": args.output,
            "method": report["method"],
            "leagues": {
                league: {
                    key: value
                    for key, value in details.items()
                    if key != "predictions"
                }
                for league, details in report["leagues"].items()
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if not args.synthetic_test_data and not args.confirm_usage_rights:
        print(
            "Training stopped: verify that this dataset permits machine-learning use, "
            "then pass --confirm-usage-rights.",
            file=sys.stderr,
        )
        return 2
    try:
        matches = load_matches(args.csv)
        reports = []
        for league in matches["league"].drop_duplicates().tolist():
            artifact = train_model(
                filter_league(matches, league),
                source_name=args.source_name,
                license_reference=args.license_reference,
                data_kind="synthetic_test" if args.synthetic_test_data else "authorized_real",
                source_collected_at=args.source_collected_at,
                source_url=args.source_url,
            )
            path = save_artifact(artifact, args.artifact_dir)
            reports.append({
                "league": league,
                "artifact": str(path),
                "model_version": artifact.model_version,
                "data_kind": artifact.source_metadata["data_kind"],
                "source_collected_at_utc": artifact.source_metadata["source_collected_at_utc"],
                "data_as_of": artifact.state.latest_date.isoformat(),
                "evaluation": asdict(artifact.evaluation),
            })
    except (CSVValidationError, ModelError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "notice": "Validation estimates only; synthetic test artifacts are explicitly labeled and are not real-match evidence.",
        "models": reports,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
