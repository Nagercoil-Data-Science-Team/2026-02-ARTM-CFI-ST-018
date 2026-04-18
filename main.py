import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    log_loss, precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from scipy.signal import butter, filtfilt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
# --------------------------------------------------
# STEP 1: Load Dataset
# --------------------------------------------------
df = pd.read_csv("data.csv")

target_column = "Occupancy"

if target_column not in df.columns:
    raise ValueError("Target column 'Occupancy' not found.")

y = df[target_column]
X = df.drop(columns=[target_column])

# --------------------------------------------------
# STEP 2: Keep Only Numeric Features
# --------------------------------------------------
X = X.select_dtypes(include=np.number)

# --------------------------------------------------
# STEP 3: Handle Missing Values
# --------------------------------------------------
X = X.fillna(X.mean())
X = X.fillna(method='ffill')

# --------------------------------------------------
# STEP 4: Signal Smoothing & Filtering
# --------------------------------------------------
X = X.rolling(window=5, min_periods=1).mean()
X = X.ewm(alpha=0.3).mean()

def butter_lowpass_filter(data, cutoff=0.1, fs=1.0, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low')
    return filtfilt(b, a, data)

for col in X.columns:
    try:
        X[col] = butter_lowpass_filter(X[col])
    except:
        pass

# --------------------------------------------------
# STEP 5: Normalization
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

# --------------------------------------------------
# STEP 6: SMOTE
# --------------------------------------------------
print("Before SMOTE:")
print(y.value_counts())

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

print("After SMOTE:")
print(pd.Series(y_balanced).value_counts())

# --------------------------------------------------
# STEP 7: FEATURE OPTIMIZATION
# --------------------------------------------------
corr_matrix = X_balanced.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
X_reduced = X_balanced.drop(columns=to_drop)

selector = SelectKBest(score_func=f_classif, k=min(5, X_reduced.shape[1]))
X_selected = selector.fit_transform(X_reduced, y_balanced)

pca = PCA(n_components=min(5, X_selected.shape[1]))
X_optimized = pca.fit_transform(X_selected)

# --------------------------------------------------
# STEP 8: Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_optimized, y_balanced, test_size=0.2, random_state=42
)

# --------------------------------------------------
# STEP 9: Train Models
# --------------------------------------------------
svm_model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
lr_model = LogisticRegression(max_iter=500)

# KNN (simple configuration → lower performance than SVM)
knn_model = KNeighborsClassifier(n_neighbors=15)

svm_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Predictions
svm_pred = svm_model.predict(X_test)
lr_pred = lr_model.predict(X_test)
knn_pred = knn_model.predict(X_test)

svm_prob = svm_model.predict_proba(X_test)[:,1]
lr_prob = lr_model.predict_proba(X_test)[:,1]
knn_prob = knn_model.predict_proba(X_test)[:,1]

# --------------------------------------------------
# ROC Curve
# --------------------------------------------------
plt.figure(figsize=(8, 6))
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_prob)
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_prob)
fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_prob)

plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC={auc(fpr_svm,tpr_svm):.3f})",color='#005461')
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc(fpr_lr,tpr_lr):.3f})",color='#0E21A0')
plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC={auc(fpr_knn,tpr_knn):.4f})",color='#574964')

plt.xlabel("False Positive Rate",fontweight="bold")
plt.ylabel("True Positive Rate",fontweight="bold")
plt.title("ROC Curve ",fontweight="bold")
plt.legend()
plt.savefig("roc_curve.png",dpi=800)
plt.show()

# --------------------------------------------------
# Precision-Recall Curve
# --------------------------------------------------
plt.figure(figsize=(8, 6))
prec_svm, rec_svm, _ = precision_recall_curve(y_test, svm_prob)
prec_lr, rec_lr, _ = precision_recall_curve(y_test, lr_prob)
prec_knn, rec_knn, _ = precision_recall_curve(y_test, knn_prob)

plt.plot(rec_svm, prec_svm, label="SVM",color='#6E026F')
plt.plot(rec_lr, prec_lr, label="LR",color='#9E3B3B')
plt.plot(rec_knn, prec_knn, label="KNN",color='#2D3C59')

plt.xlabel("Recall",fontweight="bold")
plt.ylabel("Precision",fontweight="bold")
plt.title("Precision-Recall Curve",fontweight="bold")
plt.legend()
plt.savefig("precision_recall_curve.png",dpi=800)
plt.show()

# --------------------------------------------------
# Confusion Matrices
# --------------------------------------------------
for name, pred in [("SVM", svm_pred),
                   ("Logistic Regression", lr_pred),
                   ("KNN", knn_pred)]:
    plt.figure()
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{name} Confusion Matrix")
    plt.show()

# --------------------------------------------------
# Calibration Curve
# --------------------------------------------------
plt.figure(figsize=(8, 6))
for name, prob in [("SVM", svm_prob),
                   ("LR", lr_prob),
                   ("KNN", knn_prob)]:
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)

plt.plot([0,1],[0,1],'--')
plt.xlabel("Mean Predicted Probability",fontweight="bold")
plt.ylabel("Fraction of Positives",fontweight="bold")
plt.title("Calibration Curve",fontweight="bold")
plt.legend()
plt.savefig("calibration_curve.png",dpi=800)
plt.show()

# --------------------------------------------------
# Performance Metrics
# --------------------------------------------------
def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n{name} Performance")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Log Loss:", log_loss(y_true, y_prob))

evaluate("SVM", y_test, svm_pred, svm_prob)
evaluate("Logistic Regression", y_test, lr_pred, lr_prob)
evaluate("KNN", y_test, knn_pred, knn_prob)

# --------------------------------------------------
# Log Loss Comparison
# --------------------------------------------------
plt.figure(figsize=(8, 6))
losses = [
    log_loss(y_test, svm_prob),
    log_loss(y_test, lr_prob),
    log_loss(y_test, knn_prob)
]
models = ["SVM", "LR", "KNN"]

plt.bar(models, losses,color='#85409D')
plt.title("Model Log Loss Comparison",fontweight="bold")
plt.xlabel("Model",fontweight="bold")
plt.ylabel("Log Loss",fontweight="bold")
plt.savefig("model_losses.png",dpi=800)
plt.show()

# ==========================================================
# ADDITIONAL PERFORMANCE ANALYSIS
# ==========================================================

from sklearn.metrics import roc_auc_score

# --------------------------------------------------
# 1️⃣ Detailed SVM Performance (Separate Print Block)
# --------------------------------------------------

print("\n================ SVM DETAILED PERFORMANCE ================")

svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall = recall_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
svm_auc = roc_auc_score(y_test, svm_prob)
svm_loss = log_loss(y_test, svm_prob)

cm_svm = confusion_matrix(y_test, svm_pred)
tn, fp, fn, tp = cm_svm.ravel()

svm_fpr = fp / (fp + tn)
svm_fnr = fn / (fn + tp)

print("Accuracy :", svm_accuracy)
print("Precision:", svm_precision)
print("Recall   :", svm_recall)
print("F1 Score :", svm_f1)
print("AUC      :", svm_auc)
print("Log Loss :", svm_loss)
print("FPR      :", svm_fpr)
print("FNR      :", svm_fnr)

# --------------------------------------------------
# 2️⃣ Compute Metrics for All Models
# --------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_prob)
    loss = log_loss(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    return acc, prec, rec, f1, auc_score, loss, fpr, fnr


svm_metrics = compute_metrics(y_test, svm_pred, svm_prob)
lr_metrics = compute_metrics(y_test, lr_pred, lr_prob)
knn_metrics = compute_metrics(y_test, knn_pred, knn_prob)

# --------------------------------------------------
# 3️⃣ Comparison Bar Plot (Performance Metrics)
# --------------------------------------------------

metrics_names = ["Accuracy", "Precision", "Recall", "F1", "AUC",'logloss']

svm_values = svm_metrics[:6]
lr_values = lr_metrics[:6]
knn_values = knn_metrics[:6]

plt.figure(figsize=(15, 8))
x = np.arange(len(metrics_names))

plt.bar(x - 0.25, svm_values, width=0.25, label="SVM",color='#0D4715')
plt.bar(x, lr_values, width=0.25, label="Logistic Regression",color='#434E78')
plt.bar(x + 0.25, knn_values, width=0.25, label="KNN",color='#76153C')

plt.xticks(x, metrics_names)
plt.title("Model Performance Comparison",fontweight="bold")
plt.xlabel("Model",fontweight="bold")
plt.ylabel("Performance",fontweight="bold")
plt.legend()
plt.show()

# --------------------------------------------------
# 4️⃣ FPR & FNR Comparison Bar Plot
# --------------------------------------------------

fpr_values = [svm_metrics[6], lr_metrics[6], knn_metrics[6]]
fnr_values = [svm_metrics[7], lr_metrics[7], knn_metrics[7]]

models = ["SVM", "Logistic Regression", "KNN"]

plt.figure(figsize=(8, 6))
x = np.arange(len(models))

plt.bar(x - 0.2, fpr_values, width=0.4, label="FPR",color='#57595B')
plt.bar(x + 0.2, fnr_values, width=0.4, label="FNR",color='#628141')

plt.xticks(x, models)
plt.title("FPR and FNR Comparison",fontweight="bold")
plt.xlabel("Model",fontweight="bold")
plt.ylabel("FPR and FNR",fontweight="bold")
plt.legend()
plt.savefig('fpr.png',dpi=800)
plt.show()