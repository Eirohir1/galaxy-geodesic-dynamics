import re
import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
LOG_PATH = Path("mond_geodesic_comparison_spiralonly_AIC_BIC_GPT4_output.log")

# === PARSE FUNCTION ===
def parse_fit_results(log_text):
    pattern = re.compile(
        r"Analyzing:\s+(.*?)\s+.*?"
        r"GEODESIC:\s+χ²/dof\s+=\s+([\d.]+).*?"
        r"AIC\s+=\s+([\d.]+).*?"
        r"BIC\s+=\s+([\d.]+).*?"
        r"MOND:\s+χ²/dof\s+=\s+([\d.]+).*?"
        r"AIC\s+=\s+([\d.]+).*?"
        r"BIC\s+=\s+([\d.]+)",
        re.DOTALL
    )

    entries = []
    for match in pattern.finditer(log_text):
        filename, g_chi2, g_aic, g_bic, m_chi2, m_aic, m_bic = match.groups()
        galaxy = Path(filename).stem.replace("_rotmod", "")
        entries.append({
            'Galaxy': galaxy,
            'Geodesic_Chi2_dof': float(g_chi2),
            'Geodesic_AIC': float(g_aic),
            'Geodesic_BIC': float(g_bic),
            'MOND_Chi2_dof': float(m_chi2),
            'MOND_AIC': float(m_aic),
            'MOND_BIC': float(m_bic)
        })

    return pd.DataFrame(entries)

# === CLASSIFICATION FUNCTION ===
def classify_fit_quality(chi2_dof):
    if chi2_dof < 0.8:
        return 'Overfit'
    elif chi2_dof <= 2.5:
        return 'Ideal Fit'
    else:
        return 'Underfit'

# === MAIN EXECUTION ===
if __name__ == "__main__":
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Expected log at {LOG_PATH} not found.")

    log_text = LOG_PATH.read_text()
    df = parse_fit_results(log_text)

    # Apply quality classification
    df['Geodesic_FitQuality'] = df['Geodesic_Chi2_dof'].apply(classify_fit_quality)
    df['MOND_FitQuality'] = df['MOND_Chi2_dof'].apply(classify_fit_quality)

    # Compute win indicators
    df['AIC_Winner'] = df['Geodesic_AIC'] < df['MOND_AIC']
    df['BIC_Winner'] = df['Geodesic_BIC'] < df['MOND_BIC']
    df['Overall_Winner'] = df['AIC_Winner'] & df['BIC_Winner']

    # Summary
    summary = df.groupby('Geodesic_FitQuality').size()

    # Output to terminal and optionally to CSV
    print("=== GEODESIC FIT QUALITY SUMMARY ===")
    print(summary)
    print("\n=== FULL COMPARISON TABLE ===")
    print(df.to_string(index=False))

    # Optional save
    df.to_csv("fit_quality_results.csv", index=False)
