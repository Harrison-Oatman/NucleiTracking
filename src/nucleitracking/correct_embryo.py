"""
correct_embryo.py — interactive Napari correction session for a tracked embryo.

Usage
-----
Run against a pipeline config (recommended):

    correct-embryo -c path/to/config.yml

This derives all paths from the config:
  - spots:       {dataset}/tracking_{param_set_name}/{param_set_name}_spots.h5
  - corrections: {dataset}/tracking_{param_set_name}/corrections/
  - export:      {dataset}/tracking_{param_set_name}/{param_set_name}_corrected_spots.h5

Or point directly at a spots file:

    correct-embryo --spots path/to/embryo_spots.h5

In that case corrections are stored alongside the file, in a ``corrections/``
subdirectory next to it, and the export lands next to the input file with a
``_corrected`` suffix.

Optional flags
--------------
--first-frame N
    Skip frames at or below N before opening the viewer.  Defaults to the
    value of ``local_post.tracking.start_frame`` from the config (or 0 when
    using --spots directly).
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Interactive Napari lineage correction for a tracked embryo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "-c",
        "--config",
        type=str,
        metavar="CONFIG",
        help="Path to pipeline config (.yml / .json). Derives all paths automatically.",
    )
    source.add_argument(
        "--spots",
        type=str,
        metavar="SPOTS_H5",
        help="Direct path to a *_spots.h5 file.",
    )

    parser.add_argument(
        "--first-frame",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Exclude frames at or below this index. "
            "Defaults to tracking.start_frame from config, or 0 for --spots."
        ),
    )

    args = parser.parse_args()

    # Deferred imports so --help works without local extras installed.
    from .corrections import main as run_corrections
    from .pipeline.config import PipelineConfig

    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: config file not found at {config_path}")
            return

        config = PipelineConfig.load(config_path)
        track_dir = config.dataset / f"tracking_{config.param_set_name}"
        spots_path = track_dir / f"{config.param_set_name}_spots.h5"
        corrections_dir = track_dir / "corrections"
        export_path = track_dir / f"{config.param_set_name}_corrected_spots.h5"
        first_frame = (
            args.first_frame
            if args.first_frame is not None
            else config.local_post.tracking.start_frame
        )

    else:
        spots_path = Path(args.spots)
        corrections_dir = spots_path.parent / "corrections"
        stem = spots_path.stem  # e.g. "lightsheet_20250131_spots"
        export_name = stem.replace("_spots", "_corrected_spots") + ".h5"
        export_path = spots_path.parent / export_name
        first_frame = args.first_frame if args.first_frame is not None else 0

    if not spots_path.exists():
        print(f"Error: spots file not found at {spots_path}")
        return

    print(f"Embryo:       {spots_path}")
    print(f"Corrections:  {corrections_dir}")
    print(f"Export path:  {export_path}")
    print(f"First frame:  {first_frame}")

    run_corrections(
        spots_path=spots_path,
        corrections_dir=corrections_dir,
        export_path=export_path,
        first_frame=first_frame,
    )


if __name__ == "__main__":
    main()
