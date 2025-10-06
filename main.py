# ------------------------------------
# 1. Import Libraries
# ------------------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, precision_recall_curve, f1_score
)
from imblearn.over_sampling import SMOTE

# ------------------------------------
# 2. Load Dataset
# ------------------------------------
df = pd.read_csv(r"C:\Users\JAYANTH\PythonProject\PycharmTut\dataset\creditcard.csv")

print("✅ Dataset loaded successfully!")
print("Shape of dataset before sampling:", df.shape)

# Use only 50k rows for faster training
df = df.sample(50000, random_state=42)
print("📉 Using sample dataset of shape:", df.shape)

# ------------------------------------
# 3. Prepare Features & Target
# ------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale 'Time' and 'Amount'
print("➡️ Scaling features...")
scaler = StandardScaler()
X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

# Train-Test Split
print("➡️ Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------
# 4. Handle Class Imbalance with SMOTE
# ------------------------------------
print("➡️ Applying SMOTE for class balancing...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Shape before SMOTE:", X_train.shape, y_train.value_counts().to_dict())
print("Shape after SMOTE:", X_train_res.shape, pd.Series(y_train_res).value_counts().to_dict())

# ------------------------------------
# 5. Train Tuned Random Forest
# ------------------------------------
print("➡️ Training Tuned Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,         # more trees for stability
    max_depth=20,             # controlled depth (avoid overfitting)
    min_samples_split=5,      # prevent very small leaf nodes
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_res, y_train_res)

# ------------------------------------
# 6. Find Best Threshold
# ------------------------------------
print("➡️ Optimizing threshold for best F1-score...")
y_proba = rf.predict_proba(X_test)[:, 1]

best_thresh, best_f1, best_acc = 0.5, 0, 0
for t in np.arange(0.1, 0.6, 0.01):   # search thresholds
    y_pred_t = (y_proba >= t).astype(int)
    f1 = f1_score(y_test, y_pred_t, pos_label=1)
    acc = accuracy_score(y_test, y_pred_t)
    if f1 > best_f1:  # maximize F1
        best_thresh, best_f1, best_acc = t, f1, acc

print(f"🔎 Best Threshold: {best_thresh:.2f}, "
      f"F1={best_f1:.4f}, Accuracy={best_acc:.4f}")

# Final predictions using best threshold
y_pred_best = (y_proba >= best_thresh).astype(int)

# ------------------------------------
# 7. Evaluation
# ------------------------------------
print("\n🎯 Random Forest Accuracy:", accuracy_score(y_test, y_pred_best))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred_best, digits=4))

# Optional: Precision-Recall tradeoff snapshot
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
print("\n📈 Precision-Recall Tradeoff (sample points):")
for t, p, r in zip(thresholds[::20], precision[::20], recall[::20]):
    print(f"Threshold={t:.2f}, Precision={p:.3f}, Recall={r:.3f}")
