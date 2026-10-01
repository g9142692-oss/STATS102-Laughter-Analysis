import os
import random
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    roc_curve,
    auc
)
from sklearn.ensemble import RandomForestClassifier

TRAIN_FOLDER = "files_selected"     # 30 labeled videos
FULL_FOLDER = "audio_files"         # 244 videos
HOP = 512
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Extract parameters for detection model
def extract_features(file, hop=HOP):

    y, sr = librosa.load(file, sr=None, duration=60)

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)

    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    delta_rms = np.diff(rms, prepend=rms[0])
    delta_centroid = np.diff(centroid, prepend=centroid[0])

    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    def zscore(x):
        return (x - np.mean(x)) / (np.std(x))

    return (
        zscore(rms_db),
        zscore(zcr),
        zscore(centroid),
        zscore(delta_rms),
        zscore(delta_centroid),
        zscore(flux)
    )

def binary_frames(file, annotations, hop=HOP):

    y, sr = librosa.load(file, sr=None, duration=60)
    num_frames = int(np.ceil(len(y) / hop))
    labels = np.zeros(num_frames)

    if annotations is None:
        return labels

    for start, end in annotations:
        start_frame = int(start * sr / hop)
        end_frame = int(end * sr / hop) + 1
        labels[start_frame:end_frame] = 1

    return labels


def build_dataset(files, ground_truth, folder):

    X_all = []
    y_all = []

    for file in tqdm(files, desc="Building Dataset"):
        path = os.path.join(folder, file)

        rms, zcr, centroid, delta_rms, delta_centroid, flux = extract_features(path)
        labels = binary_frames(path, ground_truth[file])

        min_len = min(len(rms), len(labels))

        features = np.vstack([
            rms[:min_len],
            zcr[:min_len],
            centroid[:min_len],
            delta_rms[:min_len],
            delta_centroid[:min_len],
            flux[:min_len]
        ]).T

        X_all.append(features)
        y_all.append(labels[:min_len])

    return np.vstack(X_all), np.hstack(y_all)

from ground_truth_dict import ground_truth

files = list(ground_truth.keys())
random.shuffle(files)

split_index = int(0.8 * len(files))
train_files = files[:split_index]
test_files = files[split_index:]

print("Train videos:", len(train_files))
print("Test videos:", len(test_files))

# Training & Evaluation: Random Forest

X_train, y_train = build_dataset(train_files, ground_truth, TRAIN_FOLDER)
X_test, y_test = build_dataset(test_files, ground_truth, TRAIN_FOLDER)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1
)

print("\n===== CROSS VALIDATION =====")

cv_auc = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

print("Best Parameters:")
print({
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 1,
    "class_weight": "balanced"
})

print("CV AUC:", round(cv_auc.mean(), 4))

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
test_auc = auc(fpr, tpr)

print("\n===== TEST RESULTS =====")
print("Test AUC:", round(test_auc, 4))
print("F1:", round(f1_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall:", round(recall_score(y_test, y_pred), 4))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred))

# Apply model to 244 videos

print("\n===== PROCESSING 244 VIDEOS =====")

video_files = [f for f in os.listdir(FULL_FOLDER) if f.endswith(".wav")]
video_files.sort()

results = []

for idx, file in enumerate(tqdm(video_files, desc="Detecting")):

    video_id = f"V{idx+1:03d}"
    path = os.path.join(FULL_FOLDER, file)

    rms, zcr, centroid, delta_rms, delta_centroid, flux = extract_features(path)

    min_len = min(len(rms), len(zcr), len(centroid),
                  len(delta_rms), len(delta_centroid), len(flux))

    X_new = np.vstack([
        rms[:min_len],
        zcr[:min_len],
        centroid[:min_len],
        delta_rms[:min_len],
        delta_centroid[:min_len],
        flux[:min_len]
    ]).T

    X_new = scaler.transform(X_new)

    preds = model.predict(X_new)

    laughter_frames = preds == 1
    non_frames = preds == 0

    frame_duration = HOP / 22050

    if np.sum(laughter_frames) > 0:

        segments = []
        start = None

        for i, val in enumerate(laughter_frames):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segments.append((start, i))
                start = None

        if start is not None:
            segments.append((start, len(laughter_frames)))

        durations = [(end - start) * frame_duration for start, end in segments]

        laugh_duration = np.mean(durations)
        laugh_volume = np.mean(rms[laughter_frames])

    else:
        laugh_duration = 0
        laugh_volume = 0

    if np.sum(non_frames) > 0:
        onset_rate = np.mean(flux[non_frames])
        expressiveness = np.var(centroid[non_frames])
        pitch_var = np.var(delta_centroid[non_frames])
        energy_var = np.var(delta_rms[non_frames])
    else:
        onset_rate = 0
        expressiveness = 0
        pitch_var = 0
        energy_var = 0

    results.append({
        "video_id": video_id,
        "laughter_duration": round(laugh_duration, 4),
        "laughter_volume": round(laugh_volume, 4),
        "onset_rate": round(onset_rate, 4),
        "expressiveness": round(expressiveness, 4),
        "pitch_variability": round(pitch_var, 4),
        "energy_variability": round(energy_var, 4)
    })

# Save csv

df = pd.DataFrame(results)
df.to_csv("laughter_analysis_results.csv", index=False)

print("\nSaved: laughter_analysis_results.csv")