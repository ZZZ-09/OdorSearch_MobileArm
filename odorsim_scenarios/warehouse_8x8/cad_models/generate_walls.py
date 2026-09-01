"""Generate an ASCII STL of a hollow 8x8x3 m warehouse room (walls only)."""
from __future__ import annotations

from pathlib import Path


def _write_face(lines: list[str], a, b, c) -> None:
    lines.append("  facet normal 0 0 0")
    lines.append("    outer loop")
    lines.append(f"      vertex {a[0]:.6f} {a[1]:.6f} {a[2]:.6f}")
    lines.append(f"      vertex {b[0]:.6f} {b[1]:.6f} {b[2]:.6f}")
    lines.append(f"      vertex {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}")
    lines.append("    endloop")
    lines.append("  endfacet")


def _box_faces(lines, x0, y0, z0, x1, y1, z1) -> None:
    p = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    for i, j, k in faces:
        _write_face(lines, p[i], p[j], p[k])


def main() -> None:
    out = Path(__file__).with_name("warehouse_8x8_walls.stl")
    lines = ["solid warehouse_8x8_walls"]
    t = 0.1
    X0, X1 = -4.0, 4.0
    Y0, Y1 = -4.0, 4.0
    Z0, Z1 = 0.0, 3.0
    _box_faces(lines, X0, Y1 - t, Z0, X1, Y1, Z1)
    _box_faces(lines, X0, Y0, Z0, X1, Y0 + t, Z1)
    _box_faces(lines, X1 - t, Y0, Z0, X1, Y1, Z1)
    _box_faces(lines, X0, Y0, Z0, X0 + t, Y1, Z1)
    lines.append("endsolid warehouse_8x8_walls")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
