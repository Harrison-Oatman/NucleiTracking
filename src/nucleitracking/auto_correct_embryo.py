"""
auto_correct_embryo.py — automatic lineage corrections for tracked embryos.

Usage
-----
    auto-correct-embryo -c path/to/config.yml [more_configs.yml ...]

For each embryo this loads the tracked spots exactly as ``correct-embryo``
does (same frame cropping and nucleus ids), replays the latest manual
corrections pickle if one exists, runs the automatic corrections in
``nucleitracking.auto_corrections`` on top, and writes:

  - corrections/corrections_<timestamp>.pkl       full action history
    (previous actions + automatic ones); opened automatically by
    ``correct-embryo`` as the starting point for manual review.
  - corrections/auto_corrections_<timestamp>.csv  one row per automatic edit,
    with the reason, frame, and position, for auditing.

Automatic actions can be undone in the viewer with Shift+Z like any other.

Optional flags
--------------
--dry-run          Only write the CSV log, not the corrections pickle.
--max-gap N        Allow orphan/terminal gap closing across up to N frames
                   (default 3; longer gaps were mostly spurious on 20250131).
--first-frame N    Override local_post.tracking.start_frame.
--replace-auto     Drop automatic actions from the latest pickle (keeping manual
                   ones) and recompute them, e.g. after the corrector changes.
"""

import argparse
import contextlib
import io
import pickle
from pathlib import Path
from time import time


def run_one(config_path: Path, args) -> None:
    import numpy as np

    from .auto_corrections import AutoCorrectParams, auto_correct
    from .corrections import DataSpace, load_spots, prepare_spots
    from .pipeline.config import PipelineConfig

    config = PipelineConfig.load(config_path)
    track_dir = config.dataset / f"tracking_{config.param_set_name}"
    spots_path = track_dir / f"{config.param_set_name}_spots.h5"
    corrections_dir = track_dir / "corrections"
    first_frame = (
        args.first_frame
        if args.first_frame is not None
        else config.local_post.tracking.start_frame
    )

    print(f"\n=== {config.param_set_name} ===")
    print(f"Spots:        {spots_path}")
    print(f"Corrections:  {corrections_dir}")
    print(f"First frame:  {first_frame}")

    _stem, _metadata, raw_df = load_spots(spots_path)
    spots_df = prepare_spots(raw_df, first_frame)

    # Replay the latest saved history so automatic edits build on manual work.
    previous = []
    existing = sorted(corrections_dir.glob("corrections_*.pkl"))
    with contextlib.redirect_stdout(io.StringIO()):
        data = DataSpace(spots_df.copy())
    if existing:
        with existing[-1].open("rb") as f:
            previous = pickle.load(f)
        if args.replace_auto:
            n_auto = sum(hasattr(a, "auto_reason") for a in previous)
            previous = [a for a in previous if not hasattr(a, "auto_reason")]
            print(f"Dropping {n_auto} earlier automatic actions")
        print(f"Replaying {len(previous)} existing actions from {existing[-1].name}")
        with contextlib.redirect_stdout(io.StringIO()):
            for action in previous:
                action.execute(data)

    params = AutoCorrectParams(max_gap=args.max_gap)
    base_df = data.df.drop(columns=["status", "track_id"])
    actions, log = auto_correct(base_df, params)

    # Self-check: replaying through the viewer's DataSpace must reproduce
    # the graph the corrector built.
    with contextlib.redirect_stdout(io.StringIO()):
        check = DataSpace(spots_df.copy())
        for action in previous + actions:
            action.execute(check)
    if not np.array_equal(
        check.df["parent_id"].to_numpy(), _final_parents(base_df, log)
    ):
        msg = "Replayed corrections do not match the corrected graph."
        raise RuntimeError(msg)

    corrections_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time())
    log_path = corrections_dir / f"auto_corrections_{timestamp}.csv"
    log.to_csv(log_path, index=False)
    print(f"Wrote {len(log)} log rows to {log_path.name}")

    if args.dry_run:
        print("Dry run: corrections pickle not written.")
        return

    pkl_path = corrections_dir / f"corrections_{timestamp}.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(previous + actions, f)
    print(
        f"Wrote {len(previous)} existing + {len(actions)} automatic actions "
        f"to {pkl_path.name}"
    )


def _final_parents(df, log):
    parents = df["parent_id"].copy()
    for row in log.itertuples():
        if row.kind == "unlink":
            parents[row.child] = -1
        elif row.kind == "link":
            parents[row.child] = row.parent
    return parents.to_numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Automatic lineage corrections for tracked embryos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="+",
        required=True,
        metavar="CONFIG",
        help="Pipeline config(s) (.yml / .json). Derives all paths automatically.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-gap", type=int, default=3, metavar="N")
    parser.add_argument("--first-frame", type=int, default=None, metavar="N")
    parser.add_argument(
        "--replace-auto",
        action="store_true",
        help="Recompute automatic actions, keeping only manual ones from the "
        "latest corrections pickle.",
    )
    args = parser.parse_args()

    for config in args.config:
        run_one(Path(config), args)


if __name__ == "__main__":
    main()
