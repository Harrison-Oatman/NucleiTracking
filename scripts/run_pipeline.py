import argparse
from pathlib import Path

from nucleitracking.pipeline.runner import PipelineRunner


def main():
    parser = argparse.ArgumentParser(description="NucleiTracking Pipeline Runner")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the pipeline configuration file (.yaml or .json)",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["local_pre", "local_post", "all"],
        default="all",
        help="Which phase of the pipeline to run.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        return

    runner = PipelineRunner(config_path)
    runner.run(phase=args.phase)


if __name__ == "__main__":
    main()
