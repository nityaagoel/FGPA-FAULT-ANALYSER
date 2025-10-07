import pandas as pd

def parse_outputs(golden_file, faulty_file):
    golden = pd.read_csv(golden_file)
    faulty = pd.read_csv(faulty_file)
    df = golden.merge(faulty, on=["a", "b"], suffixes=("_golden", "_faulty"))
    df["mismatch"] = df["sum_golden"] != df["sum_faulty"]
    df["bit_error_count"] = df.apply(
        lambda row: bin(row["sum_golden"] ^ row["sum_faulty"]).count("1"), axis=1
    )
    return df
