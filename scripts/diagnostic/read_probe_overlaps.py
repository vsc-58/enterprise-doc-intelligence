"""scripts/read_probe_overlaps.py — DIAGNOSTIC. Reads section_probe.json."""

import json
from pathlib import Path

DATA = json.loads(Path("data/processed/section_probe.json").read_text(encoding="utf-8"))
WATCH = ["Item 1", "Item 7", "Item 8"]

print("OVERLAPS")
for doc in DATA:
    for ov in doc["overlaps"]:
        print(
            f"  {doc['ticker']:<6} {ov['contained']:<8} inside {ov['contained_in']:<8} "
            f"({ov['contained_chars']:,} within {ov['container_chars']:,})"
        )

print("\nFLAGGED / ABSENT on Items 1, 7, 8")
for doc in DATA:
    for label in WATCH:
        item = doc["items"][label]
        if not item["present"]:
            print(f"  {doc['ticker']:<6} {label:<8} ABSENT  {item['error']}")
        elif item["collapsed"] or not item["head_in_raw"]:
            print(
                f"  {doc['ticker']:<6} {label:<8} {item['tokens']:>7,} tok  "
                f"share={item['share_of_filing']:.2f}  head_in_raw={item['head_in_raw']}"
            )