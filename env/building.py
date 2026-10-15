"""Building graph loaded from data/building.json: floors → nodes (stops) → 4 views each."""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class View:
    id: str                 # e.g. "F2_N05_v0"
    node_id: str
    index: int
    file: str               # relative to data/
    description: str
    candidate_spots: list[str]
    clue_surface: str | None
    low_information: bool
    people_visible: bool


@dataclass
class Node:
    id: str                 # e.g. "F2_N05"
    floor: int
    room: str
    room_type: str
    x: float
    z: float
    neighbors: list[str]
    views: list[View] = field(default_factory=list)


class Building:
    def __init__(self, path: Path = DATA_DIR / "building.json"):
        raw = json.loads(Path(path).read_text())
        self.nodes: dict[str, Node] = {}
        self.views: dict[str, View] = {}
        self.floor_names: dict[int, str] = {}
        for fl in raw["floors"]:
            self.floor_names[fl["floor"]] = fl["name"]
            for n in fl["nodes"]:
                node = Node(n["id"], fl["floor"], n["room"], n["room_type"],
                            n["position"]["x"], n["position"]["z"], list(n["neighbors"]))
                for i, v in enumerate(n["views"]):
                    view = View(v["id"], node.id, i, v["file"], v["description"], v["candidate_spots"],
                                v["clue_surface"], v["low_information"], v["people_visible"])
                    node.views.append(view)
                    self.views[view.id] = view
                self.nodes[node.id] = node
        # Stairs/elevator links between floors.
        self.vertical: dict[str, dict[int, str]] = {}
        for link in raw["vertical_links"]:
            a, b = link["from"], link["to"]
            self.vertical.setdefault(a, {})[self.nodes[b].floor] = b
            self.vertical.setdefault(b, {})[self.nodes[a].floor] = a

    @property
    def floors(self) -> list[int]:
        return sorted(self.floor_names)

    def nodes_on(self, floor: int) -> list[Node]:
        return [n for n in self.nodes.values() if n.floor == floor]

    def spot_views(self, node_id: str) -> list[View]:
        """Views at a node that show somewhere a backpack could be left."""
        return [v for v in self.nodes[node_id].views if v.candidate_spots]

    # Move costs: walking an edge = 1, changing floor = FLOOR_COST.
    FLOOR_COST = 3

    def _edges(self, node_id: str):
        for nb in self.nodes[node_id].neighbors:
            yield nb, 1
        for _, other in self.vertical.get(node_id, {}).items():
            yield other, self.FLOOR_COST

    def route(self, a: str, b: str) -> tuple[list[str], float]:
        """Cheapest route between two stops: (node path incl. both ends, cost)."""
        import heapq
        dist, prev = {a: 0}, {a: None}
        pq = [(0, a)]
        while pq:
            d, u = heapq.heappop(pq)
            if u == b:
                path = [u]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])
                return path[::-1], d
            if d > dist.get(u, 1e9):
                continue
            for v, w in self._edges(u):
                if d + w < dist.get(v, 1e9):
                    dist[v], prev[v] = d + w, u
                    heapq.heappush(pq, (d + w, v))
        raise ValueError(f"No path {a} → {b}")

    def path_cost(self, a: str, b: str) -> float:
        return self.route(a, b)[1]

    def is_connected(self) -> bool:
        start = next(iter(self.nodes))
        seen, q = {start}, deque([start])
        while q:
            u = q.popleft()
            for v, _ in self._edges(u):
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        return len(seen) == len(self.nodes)
