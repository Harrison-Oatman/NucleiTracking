"""Re-score the eval_multi masks (all models, 14 test frames) with match_score and write the
per-category points for annotate_frame.py: <model>/<E>_t<N>_cat_spots.csv / _cat_masks.csv."""

import sys
from pathlib import Path

import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from build_training_set_v2 import SETS, SPOTS
from match_score import match_score

EVAL = Path(
    r"C:\Tracking\NucleiTracking\data\interim\lightsheet\seg3d_benchmark\eval_multi"
)


def main(r=6.0):
    rows = []
    spots = {e: pd.read_hdf(SPOTS / f"{e}_spots.h5", "lineages") for e in SETS}
    for md in sorted(d for d in EVAL.iterdir() if d.is_dir()):
        for e, cfg in SETS.items():
            s_all = spots[e][~spots[e].is_interpolated.astype(bool)]
            for n in cfg["test"]:
                f = md / f"{e}_t{n:03d}_masks.tif"
                if not f.exists():
                    continue
                s = s_all[s_all.frame == n - cfg["offset"]]
                P = s[["px_z", "px_y", "px_x"]].to_numpy()
                st, cs, cm, C, labels = match_score(tifffile.imread(f), P, r)
                s.assign(category=cs)[
                    ["track_id", "px_z", "px_y", "px_x", "category"]
                ].to_csv(md / f"{e}_t{n:03d}_cat_spots.csv", index=False)
                pd.DataFrame(C, columns=["z", "y", "x"]).assign(
                    label=labels, category=cm
                ).to_csv(md / f"{e}_t{n:03d}_cat_masks.csv", index=False)
                rows.append(
                    dict(
                        model=md.name.replace("cpsam_v2_spots_", ""),
                        embryo=e,
                        n=n,
                        cycle=int(s.cycle.mode().iloc[0]),
                        **st,
                    )
                )
                print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(EVAL / "eval_matched.csv", index=False)
    print(
        d.pivot_table(
            index=["embryo", "n", "cycle"],
            columns="model",
            values=["recall_matched", "missed", "merge", "split", "extra"],
        )
        .round(3)
        .to_string()
    )
    g = (
        d.assign(oos=d.embryo != "20250131")
        .groupby(["model", "oos"])[
            ["n_spots", "matched", "merge", "missed", "split", "extra"]
        ]
        .sum()
    )
    g["recall_matched"] = g.matched / g.n_spots
    print(g.round(3).to_string())


if __name__ == "__main__":
    main()
