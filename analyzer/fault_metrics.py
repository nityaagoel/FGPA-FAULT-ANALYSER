def compute_metrics(df):
    total = len(df)
    mismatches = df["mismatch"].sum()
    coverage = ((total - mismatches) / total) * 100
    avg_bit_error = df["bit_error_count"].mean()

    return {
        "Total Test Cases": total,
        "Mismatches Detected": mismatches,
        "Fault Coverage (%)": round(coverage, 2),
        "Average Bit Error": round(avg_bit_error, 2)
    }
