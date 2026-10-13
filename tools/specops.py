"""Helpers shared by the spec-editing tools."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fit_obb_inside, obb_from_quad, obb_from_rect, canvas_quad_to_orig_pct


def box_obb(entry, angle, W, H):
    if len(entry) == 8:
        return obb_from_quad(canvas_quad_to_orig_pct(entry, W, H, angle), W, H)
    return obb_from_rect(entry, W, H, angle)


def rebase_size_labels(old_sls_polys, new_spec, W, H):
    """Rewrite derived size labels ({"of": N, ...}) of new_spec so they cover the same physical
    stretch of their box as the polygons they had before (old_sls_polys, in spec order)."""
    groups = new_spec["groups"] if "groups" in new_spec else [new_spec]
    boxes = []
    for g in groups:
        for e in g.get("box", []):
            boxes.append(fit_obb_inside(box_obb(e, g.get("angle", 0), W, H), W, H)[0])
    k = 0
    for g in groups:
        out = []
        for e in g.get("size_label", []):
            poly = old_sls_polys[k]
            k += 1
            if not isinstance(e, dict):
                out.append(e)
                continue
            A, B, _ = boxes[e["of"] - 1]
            du = np.array(B) - np.array(A)
            L2 = float(du @ du)
            ts = [float((np.array(p) - np.array(A)) @ du / L2) for p in poly]
            ent = {"of": e["of"], "t": [round(max(0.0, min(ts)), 3), round(min(1.0, max(ts)), 3)]}
            if "v" in e:
                ent["v"] = e["v"]
            out.append(ent)
        g["size_label"] = out
    return new_spec
