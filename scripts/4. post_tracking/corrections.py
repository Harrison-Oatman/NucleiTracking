"""
Legacy entry point — kept for direct execution from the scripts/ directory.

Prefer running via the package entry point instead:

    correct-embryo -c path/to/config.yml

Or for a specific spots file:

    correct-embryo --spots path/to/embryo_spots.h5
"""

from pathlib import Path

from nucleitracking.corrections import main

if __name__ == "__main__":
    # Hardcoded legacy paths from the original notebook workflow.
    # Edit these if running this file directly rather than via the CLI.
    _spots_path = Path(r"D:\Tracking\NucleiTracking\data\processed\lightsheet\spots")
    _embryo = "lightsheet_20250131_spots.h5"
    _corrections_dir = _spots_path.parent / "corrections"
    _export_path = _spots_path.parent / "lightsheet_20250131_corrected_spots.h5"

    main(
        spots_path=_spots_path / _embryo,
        corrections_dir=_corrections_dir,
        export_path=_export_path,
        first_frame=25,
    )
