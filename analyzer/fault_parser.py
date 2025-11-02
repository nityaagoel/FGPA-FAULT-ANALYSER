import pandas as pd

def parse_outputs(golden_file, faulty_file):
    golden = pd.read_csv(golden_file)
    faulty = pd.read_csv(faulty_file)
    
    # Check if data is in binary format (strings starting with 0 or 1)
    is_binary = False
    if len(golden) > 0:
        sample_val = str(golden.iloc[0]['sum'])
        is_binary = sample_val.isdigit() and all(c in '01' for c in sample_val)
    
    # Convert binary strings to integers if needed
    if is_binary:
        golden['a'] = golden['a'].apply(lambda x: int(str(x), 2))
        golden['b'] = golden['b'].apply(lambda x: int(str(x), 2))
        golden['sum'] = golden['sum'].apply(lambda x: int(str(x), 2))
        faulty['a'] = faulty['a'].apply(lambda x: int(str(x), 2))
        faulty['b'] = faulty['b'].apply(lambda x: int(str(x), 2))
        faulty['sum'] = faulty['sum'].apply(lambda x: int(str(x), 2))
    
    df = golden.merge(faulty, on=["a", "b"], suffixes=("_golden", "_faulty"))
    df["mismatch"] = df["sum_golden"] != df["sum_faulty"]
    df["bit_error_count"] = df.apply(
        lambda row: bin(row["sum_golden"] ^ row["sum_faulty"]).count("1"), axis=1
    )
    return df
