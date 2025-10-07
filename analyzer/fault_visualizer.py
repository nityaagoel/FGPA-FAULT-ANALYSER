import matplotlib.pyplot as plt

def plot_bit_errors(df):
    fig, ax = plt.subplots()
    ax.plot(df.index, df["bit_error_count"], marker="o", color="blue")
    ax.set_title("Bit Error Count per Test Case")
    ax.set_xlabel("Test Case Index")
    ax.set_ylabel("Bit Error Count")
    return fig

def plot_fault_distribution(df):
    fig, ax = plt.subplots()
    counts = df["mismatch"].value_counts()
    labels = ["Match", "Mismatch"]
    ax.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title("Fault Detection Distribution")
    return fig
