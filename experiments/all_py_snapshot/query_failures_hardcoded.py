#!/usr/bin/env python3
# query_failures_hardcoded.py — SIMBAD -> NED(Classifications) -> NED(Type)

import csv
import time
import re
from collections import Counter

from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned

# ===== HARD-CODED GALAXY NAMES =====
galaxy_names = [
    "D631-7", "DDO168", "DDO170", "ESO079-G014", "ESO116-G012", "ESO444-G084",
    "F563-1", "F568-1", "F568-3", "F574-1", "F583-1", "KK98-251", "NGC0024",
    "NGC0055", "NGC0100", "NGC0247", "NGC0289", "NGC0300", "NGC0801", "NGC1090",
    "NGC2366", "NGC2683", "NGC2915", "NGC2998", "NGC3726", "NGC3741", "NGC3769",
    "NGC3877", "NGC3893", "NGC3917", "NGC3949", "NGC4010", "NGC4051", "NGC4085",
    "NGC4088", "NGC4100", "NGC4157", "NGC4389", "NGC4559", "NGC5005", "NGC5907",
    "NGC5985", "NGC6674", "NGC7793", "NGC7814", "UGC00191", "UGC00634", "UGC00731",
    "UGC00891", "UGC01281", "UGC02455", "UGC02487", "UGC02885", "UGC04278",
    "UGC04325", "UGC04499", "UGC05716", "UGC05721", "UGC05764", "UGC05986",
    "UGC06614", "UGC06667", "UGC06917", "UGC07151", "UGC07399", "UGC07524",
    "UGC07603", "UGC08286", "UGC08490", "UGC08550", "UGC09037", "UGC11820",
    "UGC11914", "UGC12732", "UGCA442", "UGCA444"
]
# ===================================

DELAY = 0.30
OUTFILE = "failed_morphology.csv"

# --- Simple family classifier for summary ---
FAMILY_PATTERNS = [
    ("Irregular/dwarf", re.compile(r"\b(im|ib|sm|d[iI]rr|di|irr)\b", re.IGNORECASE)),
    ("Barred spiral",   re.compile(r"\bsb", re.IGNORECASE)),
    ("Spiral",          re.compile(r"\bs(?!0)\b|s[a-d]|\blate[- ]?type", re.IGNORECASE)),
    ("Early-type (S0/E)", re.compile(r"\bs0\b|\be[0-9]?", re.IGNORECASE)),
    ("Peculiar",        re.compile(r"pec", re.IGNORECASE)),
]

def classify_family(morph: str) -> str:
    if not morph or morph.lower() in {"unknown", "error"}:
        return "Unknown"
    for label, pat in FAMILY_PATTERNS:
        if pat.search(morph):
            return label
    return "Other/Composite"

# --- Query helpers ---
def simbad_morph(name: str):
    """Return (morph, note) from SIMBAD MORPHTYPE (preferred)."""
    try:
        custom = Simbad()
        custom.add_votable_fields("morphtype", "otype")
        t = custom.query_object(name)
        if t is not None and len(t) > 0:
            # Prefer MORPHTYPE if present
            if "MORPHTYPE" in t.colnames and t["MORPHTYPE"][0]:
                return str(t["MORPHTYPE"][0]), ""
            # otype can sometimes indicate class (Galaxy, BlueCompactG, etc.)
            if "OTYPE" in t.colnames and t["OTYPE"][0]:
                return str(t["OTYPE"][0]), ""
        return None, ""
    except Exception as e:
        return None, f"SIMBAD error: {e}"

def ned_class_morph(name: str):
    """Return (morph, note) from NED Classifications table."""
    try:
        cls = Ned.get_table(name, table="Classifications")
        if cls is not None and len(cls) > 0 and "Type" in cls.colnames and cls["Type"][0]:
            return str(cls["Type"][0]), ""
        return None, ""
    except Exception as e:
        return None, f"NED Classifications error: {e}"

def ned_type_morph(name: str):
    """Return (morph, note) from NED main table 'Type' (often just 'G')."""
    try:
        t = Ned.query_object(name)
        if t is not None and len(t) > 0 and "Type" in t.colnames and t["Type"][0]:
            return str(t["Type"][0]), ""
        return None, ""
    except Exception as e:
        return None, f"NED Type error: {e}"

def pick_final(simbad, ned_cls, ned_type):
    """Priority: SIMBAD->NED(Classifications)->NED(Type)->Unknown, but avoid returning bare 'G' if better exists."""
    for val in (simbad, ned_cls, ned_type):
        if val and val.strip() and val.lower() not in {"g", "galaxy"}:
            return val
    for val in (simbad, ned_cls, ned_type):
        if val and val.strip():
            return val
    return "Unknown"

def main():
    rows = []
    for i, name in enumerate(galaxy_names, 1):
        simbad_val, simbad_note = simbad_morph(name)
        nedc_val, nedc_note = (None, "")
        if not simbad_val:
            nedc_val, nedc_note = ned_class_morph(name)
        nedt_val, nedt_note = (None, "")
        if not simbad_val and not nedc_val:
            nedt_val, nedt_note = ned_type_morph(name)

        final = pick_final(simbad_val, nedc_val, nedt_val)
        fam = classify_family(final)

        note_parts = []
        for lbl, val, nt in (("SIMBAD", simbad_val, simbad_note),
                             ("NED_Class", nedc_val, nedc_note),
                             ("NED_Type", nedt_val, nedt_note)):
            if nt:
                note_parts.append(f"{lbl}: {nt}")
        note = " | ".join(note_parts) if note_parts else ""

        rows.append({
            "Galaxy": name,
            "SIMBAD_Morphology": simbad_val or "",
            "NED_Class_Morphology": nedc_val or "",
            "NED_Type": nedt_val or "",
            "Final_Morphology": final,
            "Family": fam,
            "Note": note
        })

        print(f"[{i}/{len(galaxy_names)}] {name} -> {final} "
              f"(SIMBAD={simbad_val or '-'}, NEDc={nedc_val or '-'}, NEDt={nedt_val or '-'})")
        time.sleep(DELAY)

    # Write CSV
    with open(OUTFILE, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "Galaxy","SIMBAD_Morphology","NED_Class_Morphology","NED_Type","Final_Morphology","Family","Note"
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved: {OUTFILE}")

    # Summary
    counts = Counter(r["Family"] for r in rows)
    print("\nMorphological Family Breakdown:")
    for fam, cnt in counts.most_common():
        print(f"  {fam:<22} {cnt}")

if __name__ == "__main__":
    main()
