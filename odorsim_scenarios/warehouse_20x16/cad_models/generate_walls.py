"""Generate an ASCII STL of a hollow 20x16x5 m warehouse room (walls only).

The room matches OdorSearch_MobileArm/config/warehouse.yaml bounds:
  x in [-10, 10], y in [-8, 8], z in [0, 5].
Wall thickness = 0.1 m.
"""
from __future__ import annotations

from pathlib import Path


def _write_face(lines: list[str], name: str, a: tuple[float, float, float],
                b: tuple[float, float, float], c: tuple[float, float, float]) -> None:
    """Append one triangle to an ASCII STL."""
    lines.append(f"  facet normal 0 0 0")
    lines.append("    outer loop")
    lines.append(f"      vertex {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}")
    lines.append(f"      vertex {b[0]:.6f} {b[1]:.6f} {b[2]:.6f}")
    lines.append(f"      vertex {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    lines.append("    endloop")
    lines.append("  endfacet")


def _box_faces(lines: list[str], name: str,
               x0: float, y0: float, z0: float,
               x1: float, y1: float, z1: float) -> None:
    """Append 12 triangles for an axis-aligned box."""
    p = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    # bottom / top
    _write_face(lines, name, p[0], p[2], p[1])
    _write_face(lines, name, p[0], p[3], p[2])
    _write_face(lines, name, p[4], p[5], p[6])
    _write_face(lines, name, p[4], p[6], p[7])
    # sides
    _write_face(lines, name, p[0], p[1], p[5])
    _write_face(lines, name, p[0], p[5], p[4])
    _write_face(lines, name, p[1], p[2], p[6])
    _write_face(lines, name, p[1], p[6], p[5])
    _write_face(lines, name, p[2], p[3], p[7])
    _write_face(lines, name, p[2], p[7], p[6])
    _write_face(lines, name, p[3], p[0], p[4])
    _write_face(lines, name, p[3], p[4], p[7])


def main() -> None:
    out = Path(__file__).with_name("warehouse_20x16_walls.stl")
    lines = [f"solid warehouse_20x16_walls"]

    # Wall thickness
    t = 0.1
    # Outer bounds
    X0, X1 = -10.0, 10.0
    Y0, Y1 = -8.0, 8.0
    Z0, Z1 = 0.0, 5.0

    # North wall (y = Y1)
    _box_faces(lines, "north_wall", X0, Y1 - t, Z0, X1, Y1, Z1)
    # South wall (y = Y0)
    _box_faces(lines, "south_wall", X0, Y0, Z0, X1, Y0 + t, Z1)
    # East wall (x = X1)
    _box_faces(lines, "east_wall", X1 - t, Y0, Z0, X1, Y1, Z1)
    # West wall (x = X0)
    _box_faces(lines, "west_wall", X0, Y0, Z0, X0 + t, Y1, Z1)

    lines.append("endsolid warehouse_20x16_walls")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
