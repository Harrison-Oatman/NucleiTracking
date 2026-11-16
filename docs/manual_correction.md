# Manual Lineage Correction

After running the tracking pipeline, automatic tracking errors can be reviewed
and fixed interactively using the `correct-embryo` tool. It opens a 3D Napari
viewer loaded with the tracked embryo, lets you mark and link nuclei with
keyboard shortcuts, and saves all edits as a pickle file that is automatically
replayed the next time you open the same embryo.

## Automatic corrections (run first)

Before manual review, run the automatic corrector on the tracked embryo(s):

```bash
auto-correct-embryo -c path/to/config.yml [more_configs.yml ...]
```

It loads the spots exactly as `correct-embryo` does, replays the latest
corrections pickle if there is one, and repairs the error types that dominated
the manual corrections of 20250131:

| Error | What it looks like | Fix |
|-------|--------------------|-----|
| Missed division | Mother keeps one daughter; the other starts as an orphan track | Orphan linked to the mother sitting at the daughters' midpoint |
| Stolen daughter | A 1-frame orphan spot beside the mother in anaphase captures a daughter | Daughter re-linked to the mother; spurious spot flagged `ISSUE` |
| Residual spot | Mother continues onto a spot that dies within a few frames; both daughters orphaned | Stub unlinked, daughters linked |
| Zipper shift | A row of nuclei each linked to the neighbour's descendant (terminal at one end, orphan at the other) | Region re-linked by constant-velocity assignment, scaled by the true frame interval (handles acquisition pauses) |
| Short gap | Terminal reappears as an orphan within 3 frames | Linked with flow-compensated distance |

Every link the corrector adds respects the cell cycle: a lineage may divide at
most once per cycle. Mitosis windows are detected from the division rate per
frame, and each frame belongs to the cycle span of its nearest window, so a
division in the tail of a wave still counts. A rescued daughter is rejected if
the mother's lineage already divided in that span, or if either daughter's chain
divides again within it; gap closing and zipper re-pairing apply the same rule,
and links made in the same pass are re-checked against each other before being
applied. Without this, auto-corrections produced lineages with more than 16
nuclei.

Remaining short terminal branches are then marked `ISSUE` and posterior-pole
terminals `POLE_TERMINAL`, using the same rules as **Enter** in the viewer.
Finally, a cell-cycle check logs (but does not fix) any division outside a
mitosis window or any lineage dividing twice in one cycle. These come from the
tracking itself and appear in the CSV as `double_division` /
`division_outside_mitosis` rows for review.

Output (in the `corrections/` directory):

- `corrections_<timestamp>.pkl` — previous actions + automatic ones. Opened by
  `correct-embryo` as usual; every automatic action can be undone with
  **Shift+Z**.
- `auto_corrections_<timestamp>.csv` — one row per automatic edit (step, kind,
  parent, child, frame, cost, reason, AP, x/y/z) for auditing.

Use `--replace-auto` to recompute the automatic actions after the corrector
changes: it keeps your manual actions from the latest pickle and drops the
earlier automatic ones. Use `--dry-run` to write only the CSV, and `--max-gap N` to allow gap closing
over longer gaps (off by default: on 20250131, gaps over 3 frames were mostly
spurious).

### Reviewing what changed

```bash
review-corrections -c path/to/config.yml               # highlight mode
review-corrections -c path/to/config.yml --side-by-side
```

Replays the latest corrections pickle (or `--corrections FILE`) and shows the
edits. Works for manual corrections too; edits are labelled `manual`.

- **Highlight mode:** the corrected embryo, nuclei coloured by lineage. Edited
  nuclei are ringed, new links are drawn coloured by type (zipper cyan,
  division magenta, continuation yellow, stub/head unlink orange, manual
  white), and removed links are red. `marked: issue` / `marked: pole` and
  `original tracks` are separate layers you can toggle.
- **Side-by-side:** original (left) and corrected (right) with a shared camera
  and time slider. A rescued daughter takes its mother's colour on the right.

Press **n** / **b** to jump to the next / previous edit; the camera centres on
it, and the type and reason appear in the notification panel. **m** toggles
whether ISSUE/POLE marks are included. Hover an edited nucleus to see its
`edit` and `reason` properties. Use `--no-tracks` to skip track tails if
loading is slow.

On 20250131 (validated against the manual corrections before frame 188 of the
cropped movie), ~97% of the automatic links agreed with the manual lineage, and
~80% of the manual link edits were reproduced.

---

## Launching

Run against a pipeline config (recommended):

```bash
correct-embryo -c path/to/config.yml
```

This automatically finds:
- **Spots**: `{dataset}/tracking_{param_set_name}/{param_set_name}_spots.h5`
- **Corrections**: `{dataset}/tracking_{param_set_name}/corrections/`
- **Export**: `{dataset}/tracking_{param_set_name}/{param_set_name}_corrected_spots.h5`

Or point directly at a spots file if you don't have a config:

```bash
correct-embryo --spots path/to/embryo_spots.h5
```

Optional flag — override which frames are excluded before the session starts:

```bash
correct-embryo -c config.yml --first-frame 30
```

By default this reads `local_post.tracking.start_frame` from the config (or 0
for `--spots`). Any frames at or below this index are dropped before the viewer
opens.

On startup, the most recent corrections pickle in the corrections directory is
loaded and replayed automatically, so you always resume from where you left off.

---

## Notifications

Every action logs a message to the Napari notification panel (bottom-right of
the viewer), including the nucleus ID, frame, new status, and how many actions
are currently in the history. Undo and save also log confirmation messages
there. Errors (e.g. trying to mark a non-terminal nucleus) appear as warnings
in the same panel.

## Understanding the display

Each nucleus is rendered as a sphere at its `(frame, z, y, x)` position. The
display has four color modes you can cycle through with **Q**:

| Mode | What is colored |
|------|----------------|
| **Track** (default) | Each lineage a distinct color, maintained through divisions |
| **Status** | Color by tracking role (see table below) |
| **Division count** | How many times that lineage has divided (spectral scale) |
| **Contiguity** | Lineages that have split into separate pieces, in their lineage colour; the detached nuclei outlined in red; everything else light grey |

### Contiguity mode

A lineage's nuclei normally form one patch. In each frame every nucleus is
joined to up to 6 nearest neighbours within 1.6x the median nearest-neighbour
distance; a lineage is contiguous if its members form one connected piece of
that graph, and nuclei outside their lineage's largest piece are *detached*. A detachment event is where a nucleus first becomes detached; only
events that persist for 10+ frames are listed, since short ones are mostly
sisters spreading after mitosis. A persistent detachment usually means a
mistracked link carried a nucleus (and its descendants) into the wrong lineage,
so look at the links just before the event's frame.

Press **n** / **b** to jump to the next / previous event; the nucleus is
selected and the notification gives its frame, lineage, duration and distance
from the rest of the lineage. The mode recomputes after every edit (about 2 s)
and resumes at the same position in the list.

### Status colors

| Status | Meaning |
|--------|---------|
| `START` (dark blue) | First-frame nucleus, no parents |
| `END` (dark red) | Terminal nucleus — needs review unless marked otherwise |
| `PARENT` (green) | Divided into two daughters |
| `CHILD` (tan) | Daughter of a dividing parent |
| `POLE_TERMINAL` (sage) | Terminal nucleus at the anterior or posterior pole — expected, not an error |
| `ISSUE` (pink) | Manually flagged tracking error |
| `OTHER` (grey) | Mid-track nucleus |

---

## Keyboard shortcuts

### Navigation

| Key | Action |
|-----|--------|
| **Enter** | Jump to the next unreviewed terminal nucleus. The branch back to the last division point is highlighted in cyan; the terminal itself in green. If the branch is fewer than 4 nuclei long it is auto-marked as `ISSUE`; if it is at the anterior/posterior pole (AP > 0.95) it is auto-marked as `POLE_TERMINAL`. |
| **Q** | Cycle display mode: Track → Division count → Contiguity → Status |
| **n** / **b** | Next / previous detached-lineage event (switches to Contiguity mode) |

### Editing

Select nuclei in the Napari viewer first (click to select one, Shift+click for
multiple), then press the shortcut.

| Key | Action | Notes |
|-----|--------|-------|
| **L** | Link selected nuclei as parent → child | Select two or more nuclei. They are sorted by frame; the earliest becomes the parent. All must be in strictly increasing frame order. |
| **U** | Unlink selected nucleus from its parent | Severs the parent-child relationship. The nucleus becomes a new start with no parent. If the former parent now has no children, it is added to the unmarked review queue as a new terminal. |
| **T** | Mark selected nucleus as `END` (terminal) | Only works if the nucleus has no children. |
| **P** | Mark selected nucleus as `POLE_TERMINAL` | Only works if the nucleus has no children. |
| **G** | Mark selected nucleus and its branch as `ISSUE` | Walks backward through parents until it reaches a division point or the start of the movie. |

### Saving, exporting, and undo

| Key | Action |
|-----|--------|
| **Shift+S** | Save current corrections to a timestamped pickle file in the corrections directory. Do this often. |
| **Shift+Z** | Undo the most recent action. Because actions modify the DataFrame in place, undo works by replaying all remaining actions from scratch on the original data, so it may take a moment on large datasets. |
| **Shift+E** | Export the fully corrected DataFrame to `{param_set_name}_corrected_spots.h5`. Frames are restored to their original numbering and track IDs are remapped to contiguous integers sorted by frequency. |

---

## Recommended workflow

1. Launch the viewer with `correct-embryo -c config.yml`.
2. Switch to **Status** mode (**Q** twice) so terminal nuclei appear in dark red.
3. Press **Enter** repeatedly to step through unreviewed terminals. For each one:
   - If it's a real cell death or exit: press **T** to mark as `END`.
   - If it's at a pole: press **P**.
   - If it's a tracking gap — the nucleus continues in the next frame but was not
     linked — select the terminal and its continuation, then press **L** to link
     them. Press **Enter** again to move on.
   - If the entire branch looks spurious (merge artifact, debris): press **G**.
4. Press **Shift+S** regularly to save.
5. When all terminals are resolved, press **Shift+E** to export the corrected HDF5.

The console prints how many unmarked terminals remain after each action, so you
always know how much is left.

---

## How corrections are stored

Every edit is serialized as an `Action` object and appended to a pickle file in
the corrections directory (`corrections_{timestamp}.pkl`). The original HDF5 is
never modified. On the next session the latest pickle is loaded and all actions
are replayed in order, so the data always reflects the full history of edits.

The exported `_corrected_spots.h5` is the final product intended for downstream
analysis. To write it without opening the viewer (same output as **Shift+E**):

```bash
correct-embryo -c config.yml --export-only
```
