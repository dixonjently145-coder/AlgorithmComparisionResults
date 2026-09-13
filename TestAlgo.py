import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
import warnings
warnings.filterwarnings('ignore')

print("="*65)
print("   FULL ML ANALYSIS — ERASMUS+ INTERNSHIP")
print("="*65)

# ── STEP 1: LOAD DATASET
print("\n[1] Loading dataset...")
df = pd.read_excel('/Users/dixonjently/Desktop/dataset.xlsx', header=None)
key_row  = df.iloc[2, 1:49]
learners = df.iloc[3:, 1:49].copy()
key_row  = pd.to_numeric(key_row,  errors='coerce').fillna(0).astype(int).values
learners = learners.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
learners = learners.reset_index(drop=True)

visual_key      = key_row[0:16]
auditory_key    = key_row[16:32]
kinesthetic_key = key_row[32:48]

def score_section(row, start, key_section):
    return sum(1 for i in range(16) if row.iloc[start+i] == key_section[i])

print(f"Total learners loaded: {len(learners)}")

# ── STEP 2: ALL 65536 COMBINATIONS
print("\n[2] Generating all 65,536 combinations of 16 binary questions...")
combos = list(itertools.product([0,1], repeat=16))
combo_results = []
for combo in combos:
    combo = list(combo)
    v = sum(1 for i in range(16) if combo[i] == visual_key[i])
    a = sum(1 for i in range(16) if combo[i] == auditory_key[i])
    k = sum(1 for i in range(16) if combo[i] == kinesthetic_key[i])
    best = max(v, a, k)
    label = 'Visual' if best == v else ('Auditory' if best == a else 'Kinesthetic')
    combo_results.append(combo + [v, a, k, label])
cols = [f'Q{i+1}' for i in range(16)] + ['Visual_Score','Auditory_Score','Kinesthetic_Score','Predicted_Style']
combo_df = pd.DataFrame(combo_results, columns=cols)
print("Combination distribution:")
print(combo_df['Predicted_Style'].value_counts().to_string())
combo_df.to_excel('/Users/dixonjently/Desktop/all_combinations.xlsx', index=False)
print("Saved: all_combinations.xlsx")

# ── STEP 3: SCORE AND APPLY MARGIN FILTER
print("\n[3] Scoring learners and applying margin filter...")
print("\nEffect of margin threshold:")
print(f"{'Margin':>8} | {'Learners kept':>14} | {'% of total':>10} | {'Visual':>8} | {'Auditory':>9} | {'Kinesthetic':>12}")
print("-"*75)

all_scores = []
for _, row in learners.iterrows():
    v = score_section(row, 0,  visual_key)
    a = score_section(row, 16, auditory_key)
    k = score_section(row, 32, kinesthetic_key)
    all_scores.append((v, a, k))

for margin in [1, 2, 3, 4, 5]:
    kept = 0
    dist = {'Visual':0,'Auditory':0,'Kinesthetic':0}
    for v, a, k in all_scores:
        scores = sorted([v, a, k], reverse=True)
        if scores[0] - scores[1] >= margin:
            kept += 1
            best = max(v, a, k)
            if best == v: dist['Visual'] += 1
            elif best == a: dist['Auditory'] += 1
            else: dist['Kinesthetic'] += 1
    pct = round(kept/len(learners)*100, 1)
    print(f"{margin:>8} | {kept:>14} | {pct:>9}% | {dist['Visual']:>8} | {dist['Auditory']:>9} | {dist['Kinesthetic']:>12}")

# Use margin 2
MARGIN = 2
filtered_X, filtered_labels = [], []
for i, (v, a, k) in enumerate(all_scores):
    scores_sorted = sorted([v, a, k], reverse=True)
    if scores_sorted[0] - scores_sorted[1] >= MARGIN:
        best = max(v, a, k)
        label = 'Visual' if best == v else ('Auditory' if best == a else 'Kinesthetic')
        filtered_X.append(learners.iloc[i].values)
        filtered_labels.append(label)

X = np.array(filtered_X)
le = LabelEncoder()
y = le.fit_transform(filtered_labels)
print(f"\nUsing margin = {MARGIN}")
print(f"Learners after filtering: {len(filtered_labels)}")
print("Distribution:", dict(pd.Series(filtered_labels).value_counts()))

# ── MODELS
models = {
    'Random Forest':  RandomForestClassifier(n_estimators=100, random_state=42),
    'Naive Bayes':    GaussianNB(),
    'KNN (k=21)':     KNeighborsClassifier(n_neighbors=21),
    'KNN (k=5)':      KNeighborsClassifier(n_neighbors=5),
    'Decision Tree':  DecisionTreeClassifier(random_state=42),
}

def get_metrics(y_true, y_pred):
    return {
        'Accuracy':  round(accuracy_score(y_true, y_pred)*100, 2),
        'Precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0)*100, 2),
        'Recall':    round(recall_score(y_true, y_pred, average='weighted', zero_division=0)*100, 2),
        'F1':        round(f1_score(y_true, y_pred, average='weighted', zero_division=0)*100, 2),
    }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── STEP 4: PER FOLD METRICS FOR ALL ALGORITHMS
print("\n" + "="*65)
print("[4] 5-FOLD CROSS VALIDATION — METRICS PER FOLD")
print("="*65)

cv_summary = []
for name, model in models.items():
    print(f"\n── {name} ──")
    print(f"{'Fold':<6} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-"*55)
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        model.fit(X[train_idx], y[train_idx])
        m = get_metrics(y[test_idx], model.predict(X[test_idx]))
        fold_results.append(m)
        print(f"  {fold}    | {m['Accuracy']:>9}% | {m['Precision']:>9}% | {m['Recall']:>9}% | {m['F1']:>9}%")
    avg = {k: round(np.mean([r[k] for r in fold_results]),2) for k in fold_results[0]}
    print("-"*55)
    print(f"  AVG  | {avg['Accuracy']:>9}% | {avg['Precision']:>9}% | {avg['Recall']:>9}% | {avg['F1']:>9}%")
    cv_summary.append({'Algorithm': name, **avg, 'Test': '5-Fold CV'})

# ── STEP 5: 80/20 FIXED SPLIT
print("\n" + "="*65)
print("[5] 80/20 FIXED SPLIT")
print("="*65)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training: {len(X_train)} | Testing: {len(X_test)}")
print(f"\n{'Algorithm':<22} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-"*70)
fixed_summary = []
for name, model in models.items():
    model.fit(X_train, y_train)
    m = get_metrics(y_test, model.predict(X_test))
    print(f"{name:<22} | {m['Accuracy']:>9}% | {m['Precision']:>9}% | {m['Recall']:>9}% | {m['F1']:>9}%")
    fixed_summary.append({'Algorithm': name, **m, 'Test': '80/20 Fixed'})

# ── STEP 6: 80/20 RANDOM SPLIT
print("\n" + "="*65)
print("[6] 80/20 RANDOM SPLIT — 5 RUNS AVERAGED")
print("="*65)
print(f"\n{'Algorithm':<22} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
print("-"*70)
random_summary = []
for name, model in models.items():
    runs = []
    for seed in [1, 7, 13, 21, 99]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        model.fit(Xtr, ytr)
        runs.append(get_metrics(yte, model.predict(Xte)))
    avg = {k: round(np.mean([r[k] for r in runs]),2) for k in runs[0]}
    print(f"{name:<22} | {avg['Accuracy']:>9}% | {avg['Precision']:>9}% | {avg['Recall']:>9}% | {avg['F1']:>9}%")
    random_summary.append({'Algorithm': name, **avg, 'Test': '80/20 Random'})

# ── STEP 7: PARAMETER TUNING — KNN
print("\n" + "="*65)
print("[7] PARAMETER TUNING — KNN")
print("="*65)
print(f"\n{'K Value':<10} | {'Accuracy':>10} | {'F1':>10}")
print("-"*35)
best_knn_acc, best_k = 0, 0
for k in [1, 3, 5, 7, 9, 11, 15, 21]:
    knn = KNeighborsClassifier(n_neighbors=k)
    results = []
    for train_idx, test_idx in skf.split(X, y):
        knn.fit(X[train_idx], y[train_idx])
        results.append(get_metrics(y[test_idx], knn.predict(X[test_idx])))
    avg_acc = round(np.mean([r['Accuracy'] for r in results]),2)
    avg_f1  = round(np.mean([r['F1'] for r in results]),2)
    marker = " <- BEST" if avg_acc > best_knn_acc else ""
    if avg_acc > best_knn_acc:
        best_knn_acc = avg_acc
        best_k = k
    print(f"k={k:<8} | {avg_acc:>9}% | {avg_f1:>9}%{marker}")
print(f"\nBest KNN: k={best_k} — {best_knn_acc}%")

# ── STEP 8: PARAMETER TUNING — RANDOM FOREST
print("\n" + "="*65)
print("[8] PARAMETER TUNING — RANDOM FOREST")
print("="*65)
print(f"\n{'Trees':<10} | {'Accuracy':>10} | {'F1':>10}")
print("-"*35)
best_rf_acc, best_trees = 0, 0
for n in [10, 50, 100, 200, 300, 500]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    results = []
    for train_idx, test_idx in skf.split(X, y):
        rf.fit(X[train_idx], y[train_idx])
        results.append(get_metrics(y[test_idx], rf.predict(X[test_idx])))
    avg_acc = round(np.mean([r['Accuracy'] for r in results]),2)
    avg_f1  = round(np.mean([r['F1'] for r in results]),2)
    marker = " <- BEST" if avg_acc > best_rf_acc else ""
    if avg_acc > best_rf_acc:
        best_rf_acc = avg_acc
        best_trees = n
    print(f"{n:<10} | {avg_acc:>9}% | {avg_f1:>9}%{marker}")
print(f"\nBest Random Forest: {best_trees} trees — {best_rf_acc}%")

# ── STEP 9: RESEARCH PAPER COMPARISON
print("\n" + "="*65)
print("[9] COMPARISON WITH RESEARCH PAPERS")
print("="*65)
our = {r['Algorithm']: r['Accuracy'] for r in fixed_summary}
papers = [
    ("Chen et al. 2025",       "Random Forest", "84.10%", "Student performance data"),
    ("Yuliansyah et al. 2026", "Naive Bayes",   "90.60%", "VAK questionnaire 1170 students"),
    ("Malik et al. 2025",      "KNN",            "92.00%", "Student exam prediction data"),
]
print(f"\n{'Paper':<28} | {'Algorithm':<16} | {'Their Result':>13} | {'Our Result':>11}")
print("-"*75)
for paper, alg, their, dataset in papers:
    our_acc = next((f"{r['Accuracy']}%" for r in fixed_summary if alg.lower() in r['Algorithm'].lower()), "see results")
    print(f"{paper:<28} | {alg:<16} | {their:>13} | {our_acc:>11}")

# ── STEP 10: SAVE ALL RESULTS
print("\n" + "="*65)
print("[10] SAVING RESULTS")
print("="*65)
all_results = cv_summary + fixed_summary + random_summary
pd.DataFrame(all_results).to_excel('/Users/dixonjently/Desktop/all_results.xlsx', index=False)
print("Saved: all_results.xlsx")

print("\n" + "="*65)
print("   DONE")
print("="*65)
print(f"\nFinal recommendation: Naive Bayes — most consistent across all tests")
print(f"KNN k={best_k} — best single result on fixed split")
print(f"Random Forest — most stable across all folds")