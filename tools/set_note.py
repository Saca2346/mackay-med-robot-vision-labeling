"""Set status and/or note of a label spec without touching its boxes.

Usage: python tools/set_note.py NAME [--status "selesai"] [--note "text"]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_spec, spec_groups, write_spec

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--status")
ap.add_argument("--note")
a = ap.parse_args()

spec = load_spec(a.name)
spec = {"groups": spec_groups(spec), "status": spec.get("status", "selesai"), "note": spec.get("note", "")}
if a.status:
    spec["status"] = a.status
if a.note is not None:
    spec["note"] = a.note
write_spec(a.name, spec)
print(a.name, spec["status"], "|", spec["note"])
