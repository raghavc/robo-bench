"""Ingest the Berkeley capture (IMG_8833–8900) as floors 3–6, plus the backpack answer key.

    python3 scripts/ingest_berkeley.py

Room assignment comes from capture-time bursts (4 shots ~3 s apart = one stop, four directions) and
visual inspection of every photo. Bag variants were read off full-resolution crops.
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/Users/raghav/Desktop/AirDrops/2 Floors Pictures")
DATA = ROOT / "data"

# floor → (name, style, tower_floor, [(node, room, room_type, (x, z), [(img, description, spots, clue_surface)])])
FLOORS = {
    3: ("Floor 3 — Meeting Wing", "Level 3 northeast wing: carpeted corridor, focus room and small meeting rooms", 20, [
        ("N01", "Window Walkway", "corridor", (-22, 4), [
            (8881, "Window corner with yellow lounge chairs and a small round table", ["yellow lounge chairs"], None),
            (8882, "Narrow walkway between a white column and the glass wall", [], None),
            (8883, "Glass door with a 'Level 3 Northeast' sign opening onto a carpeted corridor", [], "door sign"),
            (8884, "Long glass walkway with yellow chairs along the windows", ["yellow chairs"], None)]),
        ("N02", "Northeast Corridor", "hallway", (-10, 0), [
            (8878, "Carpeted corridor lined with wood doors and frosted-glass rooms", [], None),
            (8879, "White corridor leading to an exit door", [], None),
            (8880, "Carpeted corridor between frosted-glass meeting rooms", [], None)]),
        ("N03", "Focus Room", "focus_room", (-12, 12), [
            (8833, "Single-desk focus room: standing desk, monitor, office chair, whiteboard on the wall",
             ["under the standing desk", "office chair"], "whiteboard")]),
        ("N04", "Meeting Room A — Door Side", "meeting_room", (0, 10), [
            (8835, "Meeting room corner toward the whiteboard wall; oval table and office chairs", ["office chairs"], "whiteboard"),
            (8836, "Blank white wall with light switches", [], None),
            (8837, "Wood door with a 'thanks for using the room' sign", [], "door sign"),
            (8838, "Wall TV above the oval table; frosted glass wall on the left", ["oval table", "office chair"], None)]),
        ("N05", "Meeting Room A — Whiteboard Side", "meeting_room", (6, 14), [
            (8839, "Whiteboard covered in equations with an eraser cloth hanging from it", [], "whiteboard"),
            (8840, "Blank grey wall", [], None),
            (8841, "Looking back toward the door past the oval table", ["oval table"], None),
            (8842, "Wall TV and a whiteboard of equations; corner between a chair and the whiteboard",
             ["floor in the corner by the whiteboard", "office chair"], "whiteboard")]),
        ("N06", "Meeting Room B", "meeting_room", (2, -10), [
            (8843, "Wall TV over an oval table with four chairs and a floor fan", ["oval table", "office chairs"], None),
            (8844, "Whiteboard wall next to a frosted glass wall", [], "whiteboard"),
            (8845, "Wood door with a 'do not lock the focus room doors' sign", [], "door sign"),
            (8846, "White wall with an outlet and a power cord", [], None)]),
        ("N07", "Meeting Room C", "meeting_room", (14, -2), [
            (8847, "Mostly blank whiteboard with a speaker puck", [], "whiteboard"),
            (8848, "Frosted glass wall", [], None),
            (8849, "Door and the corner of the oval table", ["oval table"], None),
            (8850, "Wall TV over the table; corner where the TV wall meets the whiteboard wall",
             ["floor in the corner by the TV wall", "office chairs"], None)]),
    ]),
    4: ("Floor 4 — Commons", "Level 3 commons: kitchen bar and island, long tables, booths, green lounge, window seating", 30, [
        ("N01", "Commons — Northeast Entrance", "lounge", (-20, 0), [
            (8853, "Open commons: kitchen bar on the left, long white tables, wood-paneled booths", ["long tables"], None),
            (8854, "Window lounge with orange armchairs beside a TV credenza", ["orange armchairs"], None),
            (8855, "Glass doors to the 'Level 3 Northeast' corridor", [], "door sign"),
            (8856, "Wood panel wall with the kitchen counter beyond", [], None)]),
        ("N02", "Commons — Center", "lounge", (-6, 4), [
            (8857, "Booth seating next to a glass door into the corridor", ["booth"], None),
            (8858, "Window armchairs and long white tables with orange chairs", ["long tables", "orange armchairs"], None),
            (8859, "Long tables leading to the kitchen bar and a wall TV", ["long tables", "bar stools"], None),
            (8860, "Long view down the kitchen bar stools toward the green lounge", ["bar stools"], None)]),
        ("N03", "Commons — Kitchen Bar", "kitchen", (4, 10), [
            (8861, "Kitchen bar with orange stools, green tile backsplash and fridges", ["bar stools"], None),
            (8862, "Green lounge sofas across the commons", ["green sofas"], None),
            (8863, "Booth table against a cork wall", ["booth table"], None),
            (8864, "Booths and tables leading to the windows", ["booth tables"], None)]),
        ("N04", "Commons — Green Lounge", "lounge", (14, 4), [
            (8865, "Green modular sofas and an orange chair in front of a wood-slat wall with a door", ["green sofas"], None),
            (8866, "White wall and a dark corridor opening", [], None),
            (8867, "Long view of the kitchen bar and booths", ["bar stools"], None),
            (8868, "Fridges, a wall TV and a wooden bench behind green sofas", ["wooden bench", "green sofas"], None)]),
        ("N05", "Commons — Northeast Corner", "lounge", (6, -12), [
            (8885, "Long tables and the kitchen beyond a green sofa; rolling whiteboard with yellow legs", ["long tables"], "whiteboard"),
            (8886, "Green sofas with side tables in front of the fridges", ["green sofas"], None),
            (8887, "Window corner with orange armchairs and a stacking game", ["floor along the wall", "orange armchairs"], None),
            (8888, "Glass whiteboard and a rolling whiteboard of math, armchair and side tables",
             ["floor along the wall", "orange armchair"], "whiteboard")]),
        ("N06", "Commons — Kitchen Island", "kitchen", (-8, -12), [
            (8889, "Kitchen island, fridges and rows of long tables", ["long tables", "kitchen island"], None),
            (8890, "Glass whiteboard by a door, orange armchairs and side tables", ["orange armchairs"], "whiteboard"),
            (8891, "Windows with a foosball table, armchairs and a green sofa", ["armchairs", "green sofa"], None),
            (8892, "Green sofas facing the long tables and a glass whiteboard", ["green sofas"], "whiteboard")]),
    ]),
    5: ("Floor 5 — Studios", "Level 3 studio classrooms with slatted windows onto the atrium", 40, [
        ("N01", "Studio A", "classroom", (-8, 0), [
            (8870, "Studio classroom: rows of tables with black chairs, whiteboard at the front", ["tables", "chairs"], "whiteboard"),
            (8871, "Whiteboard and a wood-topped counter along the wall", ["counter"], "whiteboard"),
            (8872, "Open doorway looking out to the commons lounge", [], None),
            (8873, "Interior window with wood slats looking down into the atrium", [], None)]),
        ("N02", "Studio B", "classroom", (8, 0), [
            (8874, "Whiteboard with an attention formula above a long bench", ["bench"], "whiteboard"),
            (8875, "Glass door to a corridor, an electrical panel and a keyboard stand", [], None),
            (8876, "Tables, round café tables and a front desk with a monitor", ["tables"], None),
            (8877, "Table of framed photos, a low bench and a whiteboard; slatted window to the atrium",
             ["floor by the bench", "table"], "whiteboard")]),
    ]),
    6: ("Floor 6 — Atrium", "Atrium level: lime booths, café tables and a skylight opening", 50, [
        ("N01", "Atrium — Lime Booths", "lounge", (-8, 0), [
            (8893, "Lime-green booths and round white tables on striped green carpet", ["round tables", "booths"], None),
            (8894, "Window counter with yellow stools", ["window counter"], None),
            (8895, "Curved walkway along the glass wall", [], None),
            (8896, "Railing around the atrium skylight opening with a café table", ["café table"], None)]),
        ("N02", "Atrium — Overlook", "lounge", (8, 0), [
            (8897, "Round wood table, white chairs and lime booths", ["round table", "booths"], None),
            (8898, "Column beside curved lime booth seating", ["booth"], None),
            (8899, "White café table by the window counter", ["café table"], None),
            (8900, "Large round wood table overlooking the atrium skylight", ["round table"], None)]),
    ]),
}
EDGES = {
    3: [("N01", "N02"), ("N02", "N03"), ("N02", "N04"), ("N04", "N05"), ("N02", "N06"), ("N02", "N07")],
    4: [("N01", "N02"), ("N02", "N03"), ("N03", "N04"), ("N04", "N05"), ("N05", "N06"), ("N06", "N02"), ("N01", "N05")],
    5: [("N01", "N02")],
    6: [("N01", "N02")],
}
VERTICAL = [("F1_N01", "F3_N01", "campus walk"), ("F3_N01", "F4_N01", "glass doors"),
            ("F4_N04", "F5_N01", "studio door"), ("F5_N02", "F6_N01", "stairs to the atrium")]

# The same black Databricks backpack in four configurations.
VARIANTS = {
    "plain":        "black Databricks backpack with no cap and no keychain",
    "hat":          "black Databricks backpack with a black cap strapped on (neon-green ring logo)",
    "keychain":     "black Databricks backpack with a keychain (pink tag, keys, brass padlock)",
    "hat_keychain": "black Databricks backpack with a black cap strapped on and a keychain with a pink tag",
}
# Placements: one physical bag position, visible in one or more views. view → (variant photo, empty photo or None)
PLACEMENTS = [
    {"id": "P1", "variant": "plain", "spot": "under the standing desk", "views": {"F3_N03_v0": 8834},
     "empty": {"F3_N03_v0": 8833}, "confidence": "high"},
    {"id": "P2", "variant": "hat", "spot": "on the floor in the corner by the whiteboard", "views": {"F3_N05_v3": 8842},
     "confidence": "high", "note": "keychain not visible from this angle"},
    {"id": "P3", "variant": "hat_keychain", "spot": "on the floor in the corner by the TV wall", "views": {"F3_N07_v3": 8850},
     "confidence": "high", "alternate_photos": {"8851": "same spot, cap/keychain mostly hidden — unverified, not used"}},
    {"id": "P4", "variant": "hat_keychain", "spot": "on a bar stool at the kitchen counter",
     "views": {"F4_N03_v0": 8861, "F4_N02_v3": 8860}, "confidence": "high", "note": "small in the distance from F4_N02_v3"},
    {"id": "P5", "variant": "plain", "spot": "on the wooden bench under the wall TV", "views": {"F4_N04_v3": 8868}, "confidence": "high"},
    {"id": "P6", "variant": "hat", "spot": "on the floor against the wall between the window corner and the glass whiteboard",
     "views": {"F4_N05_v2": 8887, "F4_N05_v3": 8888}, "confidence": "high"},
    {"id": "P7", "variant": "keychain", "spot": "on an orange armchair by the glass whiteboard", "views": {"F4_N06_v1": 8890},
     "confidence": "high"},
    {"id": "P8", "variant": "keychain", "spot": "on the floor by the low bench under the whiteboard", "views": {"F5_N02_v3": 8877},
     "confidence": "high"},
    {"id": "P9", "variant": "hat", "spot": "on a round white table by the lime booths", "views": {"F6_N01_v0": 8893},
     "confidence": "high", "note": "keychain not visible from this angle"},
    # Found by the Roboflow auto-label cross-check, then confirmed on full-resolution crops.
    {"id": "P10", "variant": "hat", "spot": "on the back table by the whiteboard", "views": {"F5_N01_v0": 8870},
     "confidence": "high", "note": "small, far side of the room; keychain not visible"},
    {"id": "P11", "variant": "hat", "spot": "on a yellow lounge chair by the windows", "views": {"F3_N01_v0": 8881},
     "confidence": "medium", "note": "seen from behind in low light; cap visible, logo not"},
]


def save(img_no: int, name: str):
    im = ImageOps.exif_transpose(Image.open(SRC / f"IMG_{img_no}.JPG")).convert("RGB")
    im.thumbnail((1600, 1600))
    im.save(DATA / "images" / name, quality=88)   # re-encoded without EXIF (drops GPS)


def main():
    building = json.loads((DATA / "building.json").read_text())
    source_map = json.loads((DATA / "source_map.json").read_text())
    building["floors"] = [f for f in building["floors"] if f["floor"] < 3]
    for fid, (fname, style, tower, nodes) in FLOORS.items():
        edges = EDGES[fid]
        out_nodes = []
        for nid, room, rtype, (x, z), views in nodes:
            node_id = f"F{fid}_{nid}"
            vs = []
            for i, (img, desc, spots, clue) in enumerate(views):
                vid = f"{node_id}_v{i}"
                fname_img = f"{vid}.jpg"
                save(img, fname_img)
                source_map[fname_img] = f"IMG_{img}.JPG"
                vs.append({"id": vid, "file": f"images/{fname_img}", "source": f"IMG_{img}.JPG", "description": desc,
                           "candidate_spots": spots, "clue_surface": clue, "low_information": not spots and not clue,
                           "people_visible": False})
            nbrs = sorted({f"F{fid}_{b}" for a, b in edges if a == nid} | {f"F{fid}_{a}" for a, b in edges if b == nid})
            out_nodes.append({"id": node_id, "room": room, "room_type": rtype, "position": {"x": x, "z": z},
                              "neighbors": nbrs, "views": vs})
        building["floors"].append({"floor": fid, "name": fname, "style": style, "tower_floor": tower, "nodes": out_nodes,
                                   "capture": "UC Berkeley, Oct 15 2026"})
    building["vertical_links"] = [l for l in building["vertical_links"] if l["from"] == "F1_N01" and l["to"] == "F2_N01"]
    building["vertical_links"] += [{"from": a, "to": b, "via": via} for a, b, via in VERTICAL]
    (DATA / "building.json").write_text(json.dumps(building, indent=2))

    # Bag variant photos (the empty photo is the view's main image when one exists).
    vdir = DATA / "images" / "variants"
    vdir.mkdir(parents=True, exist_ok=True)
    for p in PLACEMENTS:
        for view, img in p["views"].items():
            if view in p.get("empty", {}):
                save(img, f"variants/{view}__bag_{p['variant']}.jpg")
                source_map[f"variants/{view}__bag_{p['variant']}.jpg"] = f"IMG_{img}.JPG"
    (DATA / "source_map.json").write_text(json.dumps(source_map, indent=1))
    (DATA / "bag_labels.json").write_text(json.dumps({
        "bag": "one physical backpack (black Databricks) photographed in four configurations",
        "variants": VARIANTS, "placements": PLACEMENTS,
        "notes": ["Views listed under a placement show that bag; every other Berkeley view shows no bag of ours.",
                  "Only F3_N03_v0 has both an empty photo (main image) and a bag photo (variants/).",
                  "IMG_8852 (Meeting Room C whiteboard with a derivative written on it) is not used."],
    }, indent=2))
    n = sum(len(nd["views"]) for f in building["floors"] if f["floor"] >= 3 for nd in f["nodes"])
    print(f"floors 3–6: {sum(len(f['nodes']) for f in building['floors'] if f['floor'] >= 3)} stops, {n} views; "
          f"{len(PLACEMENTS)} bag placements")


if __name__ == "__main__":
    sys.exit(main())
