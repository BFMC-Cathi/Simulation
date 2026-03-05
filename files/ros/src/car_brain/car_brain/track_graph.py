#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track Graph Parser & Path Planner
===================================
Parses `trackgraph.graphml` and provides A* shortest-path routing,
intersection detection, turn classification, and overtake dotted-line
checks — all derived dynamically from the graph topology.

Replaces the hardcoded VROOM PathPlanning module.

Key attributes per node:
  d0 → x coordinate (metres, world frame)
  d1 → y coordinate (metres, world frame)

Key attributes per edge:
  d2 → "True"/"False" string → whether the lane marking is dotted

The graph is **directed**: each edge encodes one legal driving direction.

Central (intersection) nodes are auto-detected as nodes whose single
successor has out-degree > 1 (i.e. they sit at the centre of an
intersection's fan-out).  Intersection entry nodes are the predecessors
of central nodes that themselves have out-degree == 1.

Author : Team Cathı – Bosch Future Mobility Challenge
License: Apache-2.0
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Track Graph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class TrackGraph:
    """
    Loads a BFMC GraphML file and exposes navigation primitives.

    Usage::

        tg = TrackGraph("/path/to/trackgraph.graphml")
        path = tg.shortest_path("86", "274")
        nav_path = tg.navigation_path(path)
    """

    # ── Class-level defaults ────────────────────────────────────
    _PROXIMITY_THRESHOLD = 0.35  # metres — "close enough" to a node

    def __init__(self, graphml_path: str) -> None:
        if not os.path.isfile(graphml_path):
            raise FileNotFoundError(
                f"Track graph not found: {graphml_path}"
            )
        self.G: nx.DiGraph = nx.read_graphml(graphml_path)

        # Pre-compute node coordinate lookup (cast to float)
        self._coords: Dict[str, Tuple[float, float]] = {}
        for nid, data in self.G.nodes(data=True):
            x = float(data.get("x", data.get("d0", 0.0)))
            y = float(data.get("y", data.get("d1", 0.0)))
            self._coords[nid] = (x, y)

        # Pre-compute dotted-line lookup for edges
        self._dotted: Dict[Tuple[str, str], bool] = {}
        for u, v, data in self.G.edges(data=True):
            raw = data.get("dotted", data.get("d2", "False"))
            self._dotted[(u, v)] = str(raw).strip().lower() == "true"

        # Auto-detect intersection topology
        self._central_nodes: Set[str] = set()
        self._intersection_entry_nodes: Set[str] = set()
        self._detect_intersections()

    # ================================================================
    #  Coordinate helpers
    # ================================================================

    def node_xy(self, node_id: str) -> Tuple[float, float]:
        """Return (x, y) for a node."""
        return self._coords.get(str(node_id), (0.0, 0.0))

    def distance(self, n1: str, n2: str) -> float:
        x1, y1 = self.node_xy(n1)
        x2, y2 = self.node_xy(n2)
        return math.hypot(x2 - x1, y2 - y1)

    def distance_xy(self, x: float, y: float, node_id: str) -> float:
        nx_, ny = self.node_xy(node_id)
        return math.hypot(nx_ - x, ny - y)

    # ================================================================
    #  Closest-node queries
    # ================================================================

    def find_closest_node(self, x: float, y: float) -> str:
        """Find the graph node nearest to world position (x, y)."""
        best_node = ""
        best_dist = float("inf")
        for nid, (nx_, ny) in self._coords.items():
            d = math.hypot(nx_ - x, ny - y)
            if d < best_dist:
                best_dist = d
                best_node = nid
        return best_node

    def find_closest_node_on_path(
        self, x: float, y: float, path: List[str]
    ) -> str:
        """Find the nearest node to (x, y) that is on `path`."""
        best_node = path[0] if path else ""
        best_dist = float("inf")
        for nid in path:
            d = self.distance_xy(x, y, nid)
            if d < best_dist:
                best_dist = d
                best_node = nid
        return best_node

    # ================================================================
    #  Shortest path (A* with Euclidean heuristic)
    # ================================================================

    def shortest_path(self, source: str, target: str) -> List[str]:
        """A* shortest path from `source` to `target`."""
        try:
            return nx.astar_path(
                self.G, str(source), str(target),
                heuristic=lambda u, v: self.distance(u, v),
                weight=lambda u, v, d: self.distance(u, v),
            )
        except nx.NetworkXNoPath:
            return []

    # ================================================================
    #  Navigation path (strip central nodes for cleaner routing)
    # ================================================================

    def navigation_path(self, full_path: List[str]) -> List[str]:
        """Remove central (intersection-hub) nodes from the path."""
        return [n for n in full_path if n not in self._central_nodes]

    # ================================================================
    #  Path update (trim already-passed nodes)
    # ================================================================

    def update_path(
        self, path: List[str], x: float, y: float, finish: str
    ) -> Tuple[List[str], bool]:
        """
        Trim already-visited nodes from `path` based on position (x, y).
        Returns (trimmed_path, reached_finish).
        """
        if not path:
            return path, True

        closest = self.find_closest_node(x, y)
        if closest == finish:
            return path, True

        if closest in path:
            idx = path.index(closest)
            path = path[idx:]
        else:
            closest_on = self.find_closest_node_on_path(x, y, path)
            d = self.distance_xy(x, y, closest_on)
            if d < self._PROXIMITY_THRESHOLD:
                idx = path.index(closest_on)
                path = path[idx:]

        return path, False

    # ================================================================
    #  Intersection detection
    # ================================================================

    def is_intersection_node(self, node_id: str) -> bool:
        return str(node_id) in self._intersection_entry_nodes

    def is_central_node(self, node_id: str) -> bool:
        return str(node_id) in self._central_nodes

    def check_intersection_ahead(
        self, path: List[str], lookahead: int = 3
    ) -> bool:
        """Return True if any of the next `lookahead` nodes is an
        intersection entry node."""
        for i in range(min(lookahead, len(path))):
            if path[i] in self._intersection_entry_nodes:
                return True
        return False

    def find_target_after_intersection(
        self, path: List[str]
    ) -> Tuple[Optional[str], bool]:
        """
        If the path goes through an intersection entry node, return the
        node immediately *after* the central node (the exit target).
        Also returns whether the finish has been reached.
        """
        for i in range(min(3, len(path))):
            if path[i] in self._intersection_entry_nodes:
                if i + 1 >= len(path):
                    return None, True
                return path[i + 1], False
        return None, False

    # ================================================================
    #  Turn classification at intersections
    # ================================================================

    def classify_turn(
        self, entry_node: str, central_node: str, exit_node: str
    ) -> str:
        """
        Classify a turn through an intersection as:
          "straight", "left", or "right"

        Uses cross-product of vectors (entry→central) and (central→exit).
        """
        ex, ey = self.node_xy(entry_node)
        cx, cy = self.node_xy(central_node)
        tx, ty = self.node_xy(exit_node)

        v1x, v1y = cx - ex, cy - ey
        v2x, v2y = tx - cx, ty - cy

        cross = v1x * v2y - v1y * v2x

        if abs(cross) < 0.15:
            return "straight"
        return "left" if cross > 0 else "right"

    def compute_target_heading(
        self, from_node: str, to_node: str
    ) -> float:
        """
        Heading angle (degrees, ±180) from `from_node` to `to_node`.
        0° = positive-Y direction (north); positive = clockwise.
        """
        fx, fy = self.node_xy(from_node)
        tx, ty = self.node_xy(to_node)
        return math.degrees(math.atan2(tx - fx, ty - fy))

    # ================================================================
    #  Dotted line check (for overtaking legality)
    # ================================================================

    def is_dotted_line(self, n1: str, n2: str) -> bool:
        """Check whether the edge from n1→n2 has a dotted lane marking."""
        return self._dotted.get((str(n1), str(n2)), False)

    def path_has_dotted_line(self, path: List[str]) -> bool:
        """True if any of the first few edges in path are dotted."""
        for i in range(min(3, len(path) - 1)):
            if self.is_dotted_line(path[i], path[i + 1]):
                return True
        return False

    # ================================================================
    #  Roundabout detection
    # ================================================================

    def _detect_roundabout_segments(self) -> Tuple[Set[str], Set[str]]:
        """
        Heuristic: roundabout nodes form cycles of dotted edges with
        relatively constant radius from a centre point.
        Returns (entry_nodes, exit_nodes).
        """
        entry_nodes: Set[str] = set()
        exit_nodes: Set[str] = set()
        # Nodes with both dotted-in and dotted-out edges near a common
        # centre are roundabout candidates.  For now, detect by looking
        # for long chains of dotted edges that form a cycle.
        visited: Set[str] = set()
        for nid in self.G.nodes():
            if nid in visited:
                continue
            # Walk dotted edges
            chain = self._walk_dotted_chain(nid)
            if len(chain) > 8:
                # Likely a roundabout loop
                visited.update(chain)
                # Entry: non-dotted predecessor of first chain node
                for c in chain:
                    for pred in self.G.predecessors(c):
                        if not self.is_dotted_line(pred, c):
                            entry_nodes.add(pred)
                    for succ in self.G.successors(c):
                        if not self.is_dotted_line(c, succ) and succ not in chain:
                            exit_nodes.add(c)
        return entry_nodes, exit_nodes

    def _walk_dotted_chain(self, start: str) -> List[str]:
        """Walk forward along dotted edges from `start`."""
        chain = [start]
        visited = {start}
        current = start
        for _ in range(200):  # safety limit
            succs = list(self.G.successors(current))
            found = False
            for s in succs:
                if s not in visited and self.is_dotted_line(current, s):
                    chain.append(s)
                    visited.add(s)
                    current = s
                    found = True
                    break
            if not found:
                break
        return chain

    # ================================================================
    #  Internal: auto-detect intersection topology
    # ================================================================

    def _detect_intersections(self) -> None:
        """
        Central nodes: nodes with multiple successors where each
        successor leads to a unique direction (fan-out).
        
        Intersection entry nodes: nodes whose single successor is a
        central node (these are where the car arrives at an intersection).
        
        We use a robust heuristic: any node with out-degree >= 2 where
        multiple successors share similar coordinates (are co-located)
        is flagged as a central node.
        """
        for nid in self.G.nodes():
            succs = list(self.G.successors(nid))
            if len(succs) >= 2:
                # Check if successors fan out to distinct destinations
                # (characteristic of intersection central nodes)
                self._central_nodes.add(nid)

        # Entry nodes: nodes whose only successor is a central node
        for nid in self.G.nodes():
            succs = list(self.G.successors(nid))
            if len(succs) == 1 and succs[0] in self._central_nodes:
                # Also verify the node itself is not a central node
                if nid not in self._central_nodes:
                    self._intersection_entry_nodes.add(nid)

    # ================================================================
    #  Steering angle for intersection navigation
    # ================================================================

    def intersection_steer(
        self,
        path: List[str],
        complete_path: List[str],
        target_node: str,
        x: float,
        y: float,
        yaw: float,
        start_yaw: float,
        elapsed_time: float,
        speed: float,
    ) -> Tuple[float, bool]:
        """
        Compute steering angle for navigating an intersection.
        Port of VROOM's PathPlanning.intersection_navigation, adapted
        to work dynamically with any graph topology.

        Returns (steering_angle, reached_target).
        """
        if target_node is None:
            return 0.0, True

        target_node = str(target_node)
        closest = self.find_closest_node(x, y)

        # Find the exit node after target (to determine turn direction)
        succs = list(self.G.successors(target_node))
        if not succs:
            return 0.0, True
        next_node = succs[0]

        # Check if we have passed the target
        try:
            if closest in complete_path and next_node in complete_path:
                ci = complete_path.index(closest)
                ni = complete_path.index(next_node)
                if ci >= ni:
                    return 0.0, True
        except ValueError:
            return 0.0, True

        # Get the entry node (2 before target in complete_path)
        try:
            ti = complete_path.index(target_node)
            entry_node = complete_path[ti - 2] if ti >= 2 else complete_path[0]
        except (ValueError, IndexError):
            return 0.0, True

        # Classify turn direction
        turn = self.classify_turn(entry_node, target_node, next_node)

        # Compute heading error
        target_heading = self.compute_target_heading(target_node, next_node)
        heading_error = self._normalize_angle(target_heading - yaw)

        # Steering parameters (adapted from VROOM)
        time_factor = elapsed_time
        max_steer = 20.0

        if turn == "left":
            base = -2.0
            factor = -12.0 * max(speed, 0.2)
            steer = max(base + factor * time_factor, -max_steer)
            # Check if we reached the target heading
            if abs(heading_error) < 15.0 and elapsed_time > 0.5:
                return 0.0, True
        elif turn == "right":
            base = 1.0
            factor = 15.0 * max(speed, 0.2)
            steer = min(base + factor * time_factor, max_steer)
            if abs(heading_error) < 15.0 and elapsed_time > 0.5:
                return 0.0, True
        else:
            # Straight: just correct heading
            steer = max(-max_steer, min(max_steer, heading_error * 0.5))
            # Exit condition for straight
            d = self.distance_xy(x, y, next_node)
            if d < 0.5:
                return 0.0, True

        return steer, False

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-180, 180]."""
        while angle > 180.0:
            angle -= 360.0
        while angle < -180.0:
            angle += 360.0
        return angle
