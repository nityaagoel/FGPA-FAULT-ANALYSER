from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def classify_faults(df):
    df["severity"] = df["bit_error_count"].apply(lambda x: 1 if x > 1 else 0)
    X = df[["bit_error_count"]]
    y = df["severity"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression().fit(X_train, y_train)
    preds = model.predict(X_test)
    return classification_report(y_test, preds, output_dict=True)
    