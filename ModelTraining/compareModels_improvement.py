import numpy as np
import pandas as pd
import time
from warnings import simplefilter

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.exceptions import ConvergenceWarning

# ignore common sklearn warnings
simplefilter(action='ignore', category=ConvergenceWarning)
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

# load data
X = pd.read_csv("feature_table.csv", header=None).to_numpy()
Y = pd.read_csv("Y.csv", header=None).to_numpy().reshape((-1,))

# use same 5 fold split for all models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fp = open("ModelComparison_submission.txt", "w")

def write_result(text=""):
    print(text, flush=True)
    fp.write(text + "\n")
    fp.flush()

results = {}

# first: logistic regression with feature selection inside pipeline
lr_pipeline = Pipeline([
    ('var', VarianceThreshold(threshold=0.0)),
    ('select', SelectKBest(mutual_info_classif, k=3500)),
    ('model', LogisticRegression(C=1, solver='lbfgs', max_iter=5000)),
])

scores = cross_validate(
    lr_pipeline,
    X,
    Y,
    cv=cv,
    scoring=['accuracy', 'balanced_accuracy'],
    n_jobs=-1
)

raw_acc = scores['test_accuracy'].mean()
bal_acc = scores['test_balanced_accuracy'].mean()
results["Pipeline LR"] = (raw_acc, bal_acc)

# second: add class balancing and scaling
balanced_lr = Pipeline([
    ('var', VarianceThreshold(threshold=0.0)),
    ('select', SelectKBest(mutual_info_classif, k=3500)),
    ('scale', MinMaxScaler()),
    ('model', LogisticRegression(
        C=0.1,
        solver='lbfgs',
        max_iter=8000,
        class_weight='balanced'
    )),
])

scores = cross_validate(
    balanced_lr,
    X,
    Y,
    cv=cv,
    scoring=['accuracy', 'balanced_accuracy'],
    n_jobs=-1
)

raw_acc = scores['test_accuracy'].mean()
bal_acc = scores['test_balanced_accuracy'].mean()
results["Balanced LR"] = (raw_acc, bal_acc)

# third: combine logistic regression and linear SVM using stacking
svm_pipeline = Pipeline([
    ('var', VarianceThreshold(threshold=0.0)),
    ('select', SelectKBest(mutual_info_classif, k=3500)),
    ('scale', MinMaxScaler()),
    ('model', CalibratedClassifierCV(
        LinearSVC(
            C=0.01,
            loss='squared_hinge',
            penalty='l2',
            max_iter=10000,
            class_weight='balanced'
        )
    )),
])

meta_model = LogisticRegression(
    C=1,
    solver='lbfgs',
    max_iter=2000,
    class_weight='balanced'
)

stack_model = StackingClassifier(
    estimators=[
        ('lr', balanced_lr),
        ('svm', svm_pipeline)
    ],
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    passthrough=False,
    n_jobs=1
)

scores = cross_validate(
    stack_model,
    X,
    Y,
    cv=cv,
    scoring=['accuracy', 'balanced_accuracy'],
    n_jobs=1
)

raw_acc = scores['test_accuracy'].mean()
bal_acc = scores['test_balanced_accuracy'].mean()
results["Stacking LR + SVM"] = (raw_acc, bal_acc)

# simple prediction speed test using the balanced logistic regression model
balanced_lr.fit(X, Y)

batch_sizes = [1, 10, 100, 1000]
speed_results = []

for size in batch_sizes:
    batch = X[:size]
    times = []

    for i in range(100):
        start = time.perf_counter()
        balanced_lr.predict(batch)
        end = time.perf_counter()
        times.append(end - start)

    median_time = np.median(times)
    avg_ms = median_time * 1000
    samples_per_sec = size / median_time

    speed_results.append((size, avg_ms, samples_per_sec))

#final results
write_result("=" * 60)
write_result("FINAL RESULTS")
write_result("=" * 60)

for name, (raw, bal) in results.items():
    write_result(f"{name:<20} Accuracy: {raw*100:.2f}%   Balanced Accuracy: {bal*100:.2f}%")

write_result()
write_result("PREDICTION SPEED")
for size, avg_ms, samples_per_sec in speed_results:
    write_result(f"Batch size {size:<4} Time: {avg_ms:.2f} ms   Samples/sec: {samples_per_sec:.0f}")

write_result()
write_result(f"Best balanced accuracy: {results['Stacking LR + SVM'][1]*100:.2f}%")

fp.close()