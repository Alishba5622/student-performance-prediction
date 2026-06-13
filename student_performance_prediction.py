# ============================================================
#   STUDENT PERFORMANCE PREDICTION SYSTEM
#   BS Artificial Intelligence — Machine Learning Lab Project
#   Concepts: Lab 01-06, Lab 11
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

print("=" * 55)
print("   STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 55)

# ============================================================
# TASK 1 — CREATE DATASET (Lab 01: Features, Labels, Data)
# ============================================================
print("\n[TASK 1] Loading Dataset...")

# Features: Study Hours, Attendance (%), Assignment Marks (/10)
# Label: Result (0 = Fail, 1 = Pass)

data = {
    "Study_Hours": [2, 3, 1, 2, 3, 1, 2, 3, 4, 2,
                    4, 5, 6, 5, 6, 4, 7, 5, 6, 4],
    "Attendance":  [50, 60, 40, 55, 62, 35, 48, 58, 68, 45,
                    70, 80, 90, 85, 88, 72, 95, 78, 82, 75],
    "Assignments": [3, 4, 2, 3, 4, 1, 2, 3, 5, 2,
                    5, 7, 8, 7, 9, 6, 10, 7, 8, 6],
    "Result":      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

X = np.array(list(zip(data["Study_Hours"], data["Attendance"], data["Assignments"])))
y = np.array(data["Result"])

print(f"\n{'#':<5} {'Study Hrs':<12} {'Attendance':<13} {'Assignments':<14} {'Result'}")
print("-" * 55)
for i in range(len(y)):
    result = "Pass" if y[i] == 1 else "Fail"
    print(f"{i+1:<5} {X[i][0]:<12} {X[i][1]:<13} {X[i][2]:<14} {result}")

print(f"\nDataset Loaded Successfully — {len(y)} records, 3 features")

# ============================================================
# TASK 2 — VISUALIZE DATASET (Charts)
# ============================================================
print("\n[TASK 2] Generating Visualizations...")

pass_idx = y == 1
fail_idx = y == 0

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Student Performance — Dataset Visualization", fontsize=14, fontweight='bold')

# Chart 1: Attendance vs Result
axes[0].bar(["Pass", "Fail"],
            [X[pass_idx, 1].mean(), X[fail_idx, 1].mean()],
            color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
axes[0].set_title("Avg Attendance % vs Result")
axes[0].set_ylabel("Attendance (%)")
axes[0].set_ylim(0, 100)
for i, v in enumerate([X[pass_idx,1].mean(), X[fail_idx,1].mean()]):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', fontweight='bold')

# Chart 2: Study Hours vs Result
axes[1].bar(["Pass", "Fail"],
            [X[pass_idx, 0].mean(), X[fail_idx, 0].mean()],
            color=["#3498db", "#e74c3c"], edgecolor="white", width=0.5)
axes[1].set_title("Avg Study Hours vs Result")
axes[1].set_ylabel("Study Hours/day")
axes[1].set_ylim(0, 8)
for i, v in enumerate([X[pass_idx,0].mean(), X[fail_idx,0].mean()]):
    axes[1].text(i, v + 0.1, f"{v:.1f} hrs", ha='center', fontweight='bold')

# Chart 3: Assignment Marks vs Result
axes[2].bar(["Pass", "Fail"],
            [X[pass_idx, 2].mean(), X[fail_idx, 2].mean()],
            color=["#9b59b6", "#e74c3c"], edgecolor="white", width=0.5)
axes[2].set_title("Avg Assignment Marks vs Result")
axes[2].set_ylabel("Marks (/10)")
axes[2].set_ylim(0, 10)
for i, v in enumerate([X[pass_idx,2].mean(), X[fail_idx,2].mean()]):
    axes[2].text(i, v + 0.1, f"{v:.1f}/10", ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("visualization.png", dpi=150, bbox_inches='tight')
plt.show()
print("Charts saved as visualization.png")

# ============================================================
# LAB 02 — HYPOTHESIS LEARNING
# ============================================================
print("\n[LAB 02] Hypothesis Learning")
print("-" * 40)
print("Hypothesis: IF Attendance > 70% AND Study Hours > 3 THEN Pass")
print()
correct_hyp = 0
for i in range(len(y)):
    pred = 1 if (X[i][1] > 70 and X[i][0] > 3) else 0
    correct_hyp += (pred == y[i])
print(f"Hypothesis Accuracy: {correct_hyp}/{len(y)} = {correct_hyp/len(y)*100:.1f}%")
print("This is a SPECIFIC hypothesis — it only fires when both conditions are met.")
print("A MORE GENERAL hypothesis would be: IF Attendance > 60% THEN Pass")

# ============================================================
# LAB 03 — BAYES THEOREM
# ============================================================
print("\n[LAB 03] Bayes Theorem — P(Pass | High Attendance)")
print("-" * 40)
high_att = X[:, 1] > 70
p_pass = y.mean()
p_high_att = high_att.mean()
p_high_att_given_pass = high_att[y == 1].mean()
p_pass_given_high_att = (p_high_att_given_pass * p_pass) / p_high_att

print(f"P(Pass)                = {p_pass:.2f}")
print(f"P(HighAtt)             = {p_high_att:.2f}")
print(f"P(HighAtt | Pass)      = {p_high_att_given_pass:.2f}")
print(f"P(Pass | HighAtt)      = {p_high_att_given_pass:.2f} × {p_pass:.2f} / {p_high_att:.2f}")
print(f"                       = {p_pass_given_high_att:.2f}  ({p_pass_given_high_att*100:.0f}%)")

# ============================================================
# TASK 3 — NAÏVE BAYES (Lab 04)
# ============================================================
print("\n[TASK 3] Naïve Bayes Classifier (Lab 04)")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_preds = nb_model.predict(X_test)
nb_acc = accuracy_score(y_test, nb_preds) * 100

print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")
print(f"\nNaïve Bayes Predictions : {['Pass' if p else 'Fail' for p in nb_preds]}")
print(f"Actual Results          : {['Pass' if p else 'Fail' for p in y_test]}")
print(f"\nNaïve Bayes Accuracy: {nb_acc:.1f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nb_preds))
print("\nClassification Report:")
print(classification_report(y_test, nb_preds, target_names=["Fail", "Pass"]))

# ============================================================
# LAB 05 — OPTIMIZATION CONCEPTS
# ============================================================
print("[LAB 05] Optimization Concepts")
print("-" * 40)
print("Optimization improves model performance by finding the best")
print("parameters (weights/probabilities) that minimize prediction error.")
print("In Naïve Bayes: parameters = class probabilities (P(feature|class))")
print("In Neural Network: parameters = weights & biases adjusted via Gradient Descent")
print("Goal: minimize a Loss Function (e.g. Cross-Entropy) → maximize accuracy\n")

# ============================================================
# LAB 06 — GRADIENT DESCENT (visualized in NN training)
# ============================================================
print("[LAB 06] Gradient Descent — shown via NN training loss curve")

# ============================================================
# TASK 4 — NEURAL NETWORK (Lab 11)
# ============================================================
print("\n[TASK 4] Neural Network (Lab 11 — Sequential Model)")
print("-" * 40)
print("Architecture:")
print("  model = Sequential()")
print("  model.add(Dense(4, activation='relu'))   # Hidden layer")
print("  model.add(Dense(1, activation='sigmoid'))# Output layer")
print()

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

nn_model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42,
    learning_rate_init=0.01
)
nn_model.fit(X_train_sc, y_train)
nn_preds = nn_model.predict(X_test_sc)
nn_acc = accuracy_score(y_test, nn_preds) * 100

print(f"Neural Network Predictions : {['Pass' if p else 'Fail' for p in nn_preds]}")
print(f"Actual Results             : {['Pass' if p else 'Fail' for p in y_test]}")
print(f"\nNeural Network Accuracy: {nn_acc:.1f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nn_preds))
print("\nClassification Report:")
print(classification_report(y_test, nn_preds, target_names=["Fail", "Pass"]))

# Gradient Descent Loss Curve
plt.figure(figsize=(8, 4))
plt.plot(nn_model.loss_curve_, color='#9b59b6', linewidth=2)
plt.title("Lab 06 — Gradient Descent: Training Loss per Epoch", fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss (Cross-Entropy)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150, bbox_inches='tight')
plt.show()
print("Loss curve saved as loss_curve.png")

# ============================================================
# TASK 5 — COMPARE BOTH MODELS
# ============================================================
print("\n[TASK 5] Model Comparison")
print("=" * 55)
print(f"  Naïve Bayes Accuracy  : {nb_acc:.1f}%")
print(f"  Neural Network Accuracy: {nn_acc:.1f}%")
print()

if nb_acc >= nn_acc:
    winner = "Naïve Bayes"
    reason = ("Naïve Bayes performed better on this small 20-record dataset.\n"
              "  It handles small data well, is fast, interpretable, and\n"
              "  avoids overfitting. Neural Networks need more data to shine.")
else:
    winner = "Neural Network"
    reason = ("Neural Network performed better by learning non-linear\n"
              "  feature interactions. With more data, this advantage grows.")

print(f"  Winner: {winner}")
print(f"  Reason: {reason}")

# Comparison bar chart
plt.figure(figsize=(6, 4))
bars = plt.bar(["Naïve Bayes", "Neural Network"],
               [nb_acc, nn_acc],
               color=["#2ecc71", "#9b59b6"], edgecolor="white", width=0.4)
for bar, acc in zip(bars, [nb_acc, nn_acc]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{acc:.1f}%", ha='center', fontweight='bold')
plt.title("Model Accuracy Comparison", fontweight='bold')
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches='tight')
plt.show()
print("Comparison chart saved as comparison.png")

# ============================================================
# PREDICT NEW STUDENT
# ============================================================
print("\n" + "=" * 55)
print("   PREDICTION FOR NEW STUDENT")
print("=" * 55)
new_student = np.array([[4, 75, 6]])  # 4 hrs, 75% att, 6/10 assignments
new_student_sc = scaler.transform(new_student)

nb_new = nb_model.predict(new_student)[0]
nn_new = nn_model.predict(new_student_sc)[0]

print(f"  Study Hours  : 4")
print(f"  Attendance   : 75%")
print(f"  Assignments  : 6/10")
print()
print(f"  Naïve Bayes  → {'PASS' if nb_new else 'FAIL'}")
print(f"  Neural Net   → {'PASS' if nn_new else 'FAIL'}")
final = "PASS" if (nb_new + nn_new) >= 1 else "FAIL"
print(f"\n  Final Prediction: *** {final} ***")
print("=" * 55)
print("\nProject Complete!")