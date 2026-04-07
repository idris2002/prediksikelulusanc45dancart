import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================
# 1. LOAD DATASET
# ==============================
df = pd.read_csv("Final_Marks_Data.csv")

print("Daftar kolom dalam dataset:")
print(df.columns)
print("Jumlah baris:", len(df))
print(df.head())

# ==============================
# 2. BUAT TARGET KELULUSAN
# ==============================
# Total nilai = IT1 + IT2 + Assignment
df["Total_Marks"] = (
    df["Internal Test 1 (out of 40)"] +
    df["Internal Test 2 (out of 40)"] +
    df["Assignment Score (out of 10)"]
)

# Lulus jika total >= 60
df["status_kelulusan"] = (df["Total_Marks"] >= 60).astype(int)

print("\nDistribusi Kelulusan:")
print(df["status_kelulusan"].value_counts())

# ==============================
# 3. FEATURE & TARGET
# ==============================
X = df[
    [
        "Attendance (%)",
        "Internal Test 1 (out of 40)",
        "Internal Test 2 (out of 40)",
        "Assignment Score (out of 10)",
        "Daily Study Hours",
    ]
]

y = df["status_kelulusan"]

# ==============================
# 4. SPLIT DATA
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData Training:", len(X_train))
print("Data Testing:", len(X_test))

# ==============================
# 5. MODEL C4.5
# ==============================
model_c45 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    min_samples_split=10,
    random_state=42,
)

model_c45.fit(X_train, y_train)  # WAJIB ADA

# ==============================
# 6. MODEL CART
# ==============================
model_cart = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=10,
    random_state=42,
)

model_cart.fit(X_train, y_train)  # WAJIB ADA

# ==============================
# 7. EVALUASI
# ==============================
pred_c45 = model_c45.predict(X_test)
pred_cart = model_cart.predict(X_test)

print("\n=== C4.5 ===")
print("Accuracy:", accuracy_score(y_test, pred_c45))
print(confusion_matrix(y_test, pred_c45))
print(classification_report(y_test, pred_c45))

print("\n=== CART ===")
print("Accuracy:", accuracy_score(y_test, pred_cart))
print(confusion_matrix(y_test, pred_cart))
print(classification_report(y_test, pred_cart))

# ==============================
# 8. SAVE MODEL
# ==============================
joblib.dump(model_c45, "model_c45.pkl")
joblib.dump(model_cart, "model_cart.pkl")

print("\nModel berhasil disimpan TANPA ERROR.")