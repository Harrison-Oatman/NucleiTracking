# Manual Lineage Correction

After running the tracking pipeline, automatic tracking errors can be reviewed
and fixed interactively using the `correct-embryo` tool. It opens a 3D Napari
viewer loaded with the tracked embryo, lets you mark and link nuclei with
keyboard shortcuts, and saves all edits as a pickle file that is automatically
replayed the next time you open the same embryo.

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
display has three color modes you can cycle through with **Q**:

| Mode | What is colored |
|------|----------------|
| **Track** (default) | Each lineage a distinct color, maintained through divisions |
| **Status** | Color by tracking role (see table below) |
| **Division count** | How many times that lineage has divided (spectral scale) |

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
| **Q** | Cycle display mode: Track → Status → Division count |

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
analysis.
