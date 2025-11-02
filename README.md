# 🔍 FPGA Fault Analyzer Dashboard

A comprehensive web-based tool for detecting and analyzing faults in FPGA (Field-Programmable Gate Array) simulation outputs. The system compares expected (golden) outputs with actual (faulty) outputs to calculate bit errors and automatically classify fault severity using machine learning.

## 📋 Table of Contents

- [Overview](#overview)
- [How Bit Error is Calculated](#how-bit-error-is-calculated)
- [Machine Learning for Fault Classification](#machine-learning-for-fault-classification)
- [Complete Workflow Example](#complete-workflow-example)
- [Installation](#installation)
- [Usage](#usage)

---

## 🎯 Overview

This project performs **two distinct operations**:

1. **Bit Error Calculation** (Mathematical/Direct): Uses XOR operation to count differing bits
2. **Fault Severity Classification** (Machine Learning): Uses Logistic Regression to classify fault severity based on bit error counts

**Important**: Machine Learning does NOT calculate bit errors. Instead, ML classifies the severity of bit errors that are already calculated using mathematical operations.

---

## 🔬 How Bit Error is Calculated

### Mathematical Method (NOT Machine Learning)

Bit errors are calculated using **XOR (exclusive OR)** operation, which is a deterministic mathematical process:

#### Step 1: XOR Operation
Compare golden (expected) and faulty (actual) outputs using XOR:
```
Bit Error = golden_output XOR faulty_output
```

#### Step 2: Count '1' Bits
Count the number of '1' bits in the XOR result (Hamming Distance):
```
Bit Error Count = Number of '1' bits in XOR result
```

### Example Calculation

#### Example 1: Single Bit Error

**Input Data:**
```csv
# Golden Output (Expected)
a=00000000, b=00000010, sum=0000000010  (Decimal: 0+2=2)

# Faulty Output (Actual)
a=00000000, b=00000010, sum=0000001010  (Decimal: 0+2=10)
```

**Calculation Process:**

1. **Convert to Decimal:**
   - Golden sum: `0000000010` (binary) = `2` (decimal)
   - Faulty sum: `0000001010` (binary) = `10` (decimal)

2. **XOR Operation:**
   ```
   Golden:  0000000010  (2 in decimal)
   Faulty:  0000001010  (10 in decimal)
   XOR:     0000001000  (8 in decimal)
   ```

3. **Count '1' Bits:**
   ```
   XOR result: 0000001000
   Bit positions: 876543210
                     ↑
   Only position 3 has '1'
   Bit Error Count = 1
   ```

4. **Interpretation:**
   - Bit error count = **1** (single bit flip at position 3)
   - Mismatch = **True**

#### Example 2: Multi-Bit Error

**Input Data:**
```csv
# Golden Output
a=00000100, b=00000000, sum=0000000100  (Decimal: 4+0=4)

# Faulty Output
a=00000100, b=00000000, sum=1000100100  (Decimal: 4+0=548)
```

**Calculation Process:**

1. **XOR Operation:**
   ```
   Golden:  0000000100  (4 in decimal)
   Faulty:  1000100100  (548 in decimal)
   XOR:     1000100000  (544 in decimal)
   ```

2. **Count '1' Bits:**
   ```
   XOR result: 1000100000
   Bit positions: 9876543210
                  ↑   ↑
   Positions 9 and 5 have '1'
   Bit Error Count = 2
   ```

3. **Interpretation:**
   - Bit error count = **2** (two bit flips)
   - Mismatch = **True**

#### Example 3: No Error

**Input Data:**
```csv
# Golden Output
a=00000010, b=00000010, sum=0000000100  (Decimal: 2+2=4)

# Faulty Output
a=00000010, b=00000010, sum=0000000100  (Decimal: 2+2=4)
```

**Calculation Process:**

1. **XOR Operation:**
   ```
   Golden:  0000000100  (4 in decimal)
   Faulty:  0000000100  (4 in decimal)
   XOR:     0000000000  (0 in decimal)
   ```

2. **Count '1' Bits:**
   ```
   XOR result: 0000000000
   No '1' bits present
   Bit Error Count = 0
   ```

3. **Interpretation:**
   - Bit error count = **0** (no errors)
   - Mismatch = **False**

### Code Implementation

```python
# From analyzer/fault_parser.py

def parse_outputs(golden_file, faulty_file):
    # Read CSV files
    golden = pd.read_csv(golden_file)
    faulty = pd.read_csv(faulty_file)
    
    # Convert binary to decimal if needed
    if is_binary_format:
        golden['sum'] = golden['sum'].apply(lambda x: int(str(x), 2))
        faulty['sum'] = faulty['sum'].apply(lambda x: int(str(x), 2))
    
    # Merge dataframes
    df = golden.merge(faulty, on=["a", "b"], suffixes=("_golden", "_faulty"))
    
    # Calculate bit error using XOR
    df["bit_error_count"] = df.apply(
        lambda row: bin(row["sum_golden"] ^ row["sum_faulty"]).count("1"), 
        axis=1
    )
    
    return df
```

**Key Points:**
- ✅ **Deterministic**: Same inputs always produce same bit error count
- ✅ **Mathematical**: Based on XOR operation, not learning
- ✅ **Real-time**: Calculated directly from comparison
- ✅ **No training required**: Works immediately

---

## 🤖 Machine Learning for Fault Classification

### Purpose: Classify Fault Severity (NOT Calculate Bit Errors)

After bit errors are calculated, **Machine Learning is used to classify fault severity** into categories:

- **Class 0 (Minor Fault)**: bit_error_count ≤ 1
- **Class 1 (Severe Fault)**: bit_error_count > 1

### Why Use ML for Classification?

1. **Automation**: Automatically categorize large datasets
2. **Scalability**: Handles growing test suites efficiently
3. **Pattern Learning**: Can learn complex relationships (with more features)
4. **Extensibility**: Easy to add more severity levels

### ML Workflow

#### Step 1: Prepare Training Data

**Input Feature:**
- `bit_error_count`: Already calculated bit error count

**Target Label:**
```python
severity = 1 if bit_error_count > 1 else 0
```

#### Step 2: Train-Test Split

```python
# 70% for training, 30% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
```

#### Step 3: Train Model

```python
# Logistic Regression for binary classification
model = LogisticRegression().fit(X_train, y_train)
```

#### Step 4: Predict & Evaluate

```python
predictions = model.predict(X_test)
report = classification_report(y_test, predictions)
```

### Complete Example: From Bit Error to ML Classification

#### Step 1: Calculate Bit Errors (Mathematical)

**Input Dataset:**
```csv
Test Case | Golden Sum | Faulty Sum | XOR Result | Bit Error Count
----------|------------|------------|------------|----------------
1         | 2 (0010)   | 10 (1010)  | 1000       | 1
2         | 4 (0100)   | 548 (1000100100) | 1000100000 | 2
3         | 7 (0111)   | 4 (0100)   | 0011       | 2
4         | 5 (0101)   | 5 (0101)   | 0000       | 0
5         | 12 (1100)  | 13 (1101)  | 0001       | 1
```

#### Step 2: Create Labels for ML

```python
Test Case | Bit Error Count | Severity Label | Classification
----------|-----------------|----------------|---------------
1         | 1               | 0              | Minor (≤1)
2         | 2               | 1              | Severe (>1)
3         | 2               | 1              | Severe (>1)
4         | 0               | 0              | Minor (≤1)
5         | 1               | 0              | Minor (≤1)
```

#### Step 3: Train ML Model

**Training Data (70%):**
```python
X_train = [[1], [2], [2], [0]]  # Bit error counts
y_train = [0, 1, 1, 0]           # Severity labels
```

**Model Training:**
```python
model = LogisticRegression()
model.fit(X_train, y_train)

# Model learns:
# - If bit_error_count ≤ 1 → Predict Class 0 (Minor)
# - If bit_error_count > 1 → Predict Class 1 (Severe)
```

#### Step 4: Test ML Model

**Test Data (30%):**
```python
X_test = [[1]]   # Bit error count = 1
y_test = [0]     # Actual label = Minor

# Prediction
prediction = model.predict([[1]])  # Output: [0] = Minor ✓
```

#### Step 5: ML Model Output

```python
{
  "0": {
    "precision": 0.95,  # When predicting Minor, 95% correct
    "recall": 0.98,     # Captures 98% of actual Minor faults
    "f1-score": 0.96,
    "support": 150
  },
  "1": {
    "precision": 0.92,  # When predicting Severe, 92% correct
    "recall": 0.85,     # Captures 85% of actual Severe faults
    "f1-score": 0.88,
    "support": 50
  },
  "accuracy": 0.94     # Overall 94% classification accuracy
}
```

### Code Implementation

```python
# From analyzer/fault_classifier.py

def classify_faults(df):
    # Create severity labels based on bit_error_count
    df["severity"] = df["bit_error_count"].apply(
        lambda x: 1 if x > 1 else 0
    )
    
    # Prepare features and labels
    X = df[["bit_error_count"]]  # Feature: pre-calculated bit error
    y = df["severity"]            # Label: minor (0) or severe (1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = LogisticRegression().fit(X_train, y_train)
    
    # Predict
    preds = model.predict(X_test)
    
    # Evaluate
    return classification_report(y_test, preds, output_dict=True)
```

---

## 🔄 Complete Workflow Example

### End-to-End Process

#### Input Files

**golden_output.csv (Binary Format):**
```csv
a,b,sum
00000000,00000010,0000000010
00000100,00000000,0000000100
00000011,00000100,0000000111
```

**faulty_output.csv (Binary Format):**
```csv
a,b,sum
00000000,00000010,0000001010
00000100,00000000,1000100100
00000011,00000100,0000000111
```

#### Step 1: Parse & Calculate Bit Errors

```python
# Automated by fault_parser.py

Test Case 1:
  Golden: 0000000010 (2)  XOR  Faulty: 0000001010 (10)
  XOR: 0000001000 (8)  →  Bit Error Count = 1

Test Case 2:
  Golden: 0000000100 (4)  XOR  Faulty: 1000100100 (548)
  XOR: 1000100000 (544)  →  Bit Error Count = 2

Test Case 3:
  Golden: 0000000111 (7)  XOR  Faulty: 0000000111 (7)
  XOR: 0000000000 (0)  →  Bit Error Count = 0
```

**Result DataFrame:**
```python
   a  b  sum_golden  sum_faulty  mismatch  bit_error_count
0  0  2           2          10      True                1
1  4  0           4         548      True                2
2  3  4           7           7     False                0
```

#### Step 2: Compute Metrics

```python
Total Test Cases: 3
Mismatches Detected: 2
Fault Coverage: 33.33%
Average Bit Error: 1.0
```

#### Step 3: ML Classification

**Input to ML:**
```python
bit_error_count | severity_label
----------------|---------------
1               | 0 (Minor)
2               | 1 (Severe)
0               | 0 (Minor)
```

**ML Model Predictions:**
```python
Test Case 1: bit_error=1 → Predict "Minor" (Class 0) ✓
Test Case 2: bit_error=2 → Predict "Severe" (Class 1) ✓
Test Case 3: bit_error=0 → Predict "Minor" (Class 0) ✓
```

**ML Report:**
```json
{
  "accuracy": 1.0,
  "0": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0},
  "1": {"precision": 1.0, "recall": 1.0, "f1-score": 1.0}
}
```

---

## ⚡ Key Distinction

| Aspect | Bit Error Calculation | Machine Learning |
|--------|----------------------|------------------|
| **Purpose** | Count differing bits | Classify fault severity |
| **Method** | XOR operation (mathematical) | Logistic Regression (ML) |
| **Input** | Golden + Faulty outputs | Bit error counts |
| **Output** | Bit error count (number) | Severity class (0 or 1) |
| **Requires Training** | ❌ No | ✅ Yes |
| **Deterministic** | ✅ Yes | ⚠️ Probabilistic |

### Summary

1. **Bit Error Calculation** = Mathematical XOR operation → Direct calculation
2. **Machine Learning** = Uses calculated bit errors → Classifies severity

**ML does NOT determine bit errors** - it classifies the severity of bit errors that are already calculated!

---

## 🚀 Installation

### Prerequisites
- Python 3.7+
- pip

### Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `streamlit` - Web dashboard framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Data visualization
- `scikit-learn` - Machine learning

---

## 📱 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Dashboard

1. **Upload Files:**
   - Upload `golden_output.csv` (expected outputs)
   - Upload `faulty_output.csv` (actual/test outputs)

2. **View Results:**
   - **Metrics**: Total test cases, mismatches, fault coverage, average bit error
   - **Visualizations**: Bit error trends, fault distribution
   - **ML Report**: Fault severity classification with precision/recall metrics

### File Format

**Supported Formats:**
- **Binary Format** (Recommended for FPGA):
  ```csv
  a,b,sum
  00000000,00000010,0000000010
  00000100,00000000,0000000100
  ```
  - `a`, `b`: 8-bit binary inputs
  - `sum`: 10-bit binary output

- **Decimal Format**:
  ```csv
  a,b,sum
  0,2,2
  4,0,4
  ```

The system automatically detects and handles both formats!

---

## 📊 Example Output

### Dashboard Metrics

```
Total Test Cases: 300
Mismatches Detected: 50
Fault Coverage: 83.33%
Average Bit Error: 1.2
```

### ML Classification Report

```json
{
  "0": {
    "precision": 0.95,
    "recall": 0.98,
    "f1-score": 0.96,
    "support": 250
  },
  "1": {
    "precision": 0.92,
    "recall": 0.85,
    "f1-score": 0.88,
    "support": 50
  },
  "accuracy": 0.94
}
```

---

## 🎯 Use Cases

- **FPGA Verification**: Validate FPGA designs by comparing simulation outputs
- **Fault Injection Testing**: Test fault tolerance and error detection capabilities
- **Quality Assurance**: Automate regression testing for digital circuits
- **Research**: Study fault propagation and reliability metrics

---

## 📝 License

This project is open source and available for educational and research purposes.
