# Data room

Two real office floors captured as a Street-View-style graph: 16 stops × 4 views = 64 photos.

| Path | What it is |
|---|---|
| `images/F{floor}_N{stop}_v{view}.jpg` | Photos, auto-rotated, 1600 px, EXIF (incl. GPS) stripped |
| `source_map.json` | Renamed file → original `IMG_*.JPG` |
| `building.json` | Floors, stops, room names, view descriptions, candidate spots, clue surfaces, edges |
| `annotations/anchors.json` | Per-view object labels from Roboflow auto-label (Gemini 4 Argon, 13 classes, 530 boxes) |
| `scenarios/sample_*.json` | One generated scenario per difficulty (hidden timeline + clues + placement) |
| `capture_plan.md` | Which views to reshoot with the target / decoy backpack |
| `images/variants/` | Target/decoy reshoots go here (`<view>__target.jpg`, `<view>__decoy.jpg`) |

Roboflow project: `raghavs-workspace-biwvk/office-search-env` (batches `floor-1-baseline`, `floor-2-baseline`).

Notes
- Edges and stop positions in `building.json` are inferred from the photos; verify against the real layout.
- Four views already contain real bags belonging to people in the photos (`F1_N06_v1`, `F2_N05_v0`,
  `F2_N06_v2`, `F2_N07_v3`). They are kept as natural decoys.
- Eight views show people (flagged `people_visible`). Blur faces before publishing anything publicly.
