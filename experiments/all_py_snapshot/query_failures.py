import json
import csv
from astroquery.ipac.ned import Ned
from astropy.table import Table

# === Load failed galaxy list ===
with open("results.json", "r") as f:
    data = json.load(f)

# Assumes your JSON has structure with "failures" list:
failed_galaxies = data.get("failures", [])
if not failed_galaxies:
    raise ValueError("No failed galaxies found in results.json")

# === Query NED and extract morphology ===
output_rows = []
for name in failed_galaxies:
    try:
        result = Ned.query_object(name)
        if len(result) == 0:
            morphology = "Not found"
        else:
            morphology = result["Type"][0] if "Type" in result.colnames else "Unknown"
    except Exception as e:
        morphology = f"Error: {e}"
    
    output_rows.append({
        "Galaxy": name,
        "Morphology": morphology,
        "FailureReason": "Fit failed"  # Replace if you have specific failure reasons
    })

# === Save to CSV ===
with open("failed_morphology.csv", "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["Galaxy", "Morphology", "FailureReason"])
    writer.writeheader()
    writer.writerows(output_rows)

print("Saved morphology data to failed_morphology.csv")
