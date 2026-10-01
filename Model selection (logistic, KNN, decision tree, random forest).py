import librosa
import numpy as np
import os
import random
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    roc_curve,
    auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

'''Build a dictionary of laughter segments (seconds) in the first minute of the 30 recordings'''
ground_truth = {

"20250620_How I Bagged My Husband： Gina Brillon.wav":
[(5,6),(24,30),(41,42),(46,47),(50,52),(56,59)],

"20250622_45 Minutes of Greg Warren ｜ The Salesman.wav":
[(12,14),(20,21),(31,33),(36,37),(39,51)],

"20250702_How I Got Fired At Target： Ryan Erwin.wav":
[(9,13),(24,26),(34,37)],

"20250707_The Truth about Kids： Raanan Hershberg.wav":
[(24,26),(29,30),(49,51),(59,60)],

"20250724_Teri Hatcher on Plastic Surgery & Aging.wav":
[(5,7),(9,10),(19,22),(42,43),(55,60)],

"20250730_Married Life According to Larry the Cable Guy.wav":
[(16,19),(23,25),(48,55),(58,60)],

"20250801_Jogging Was Getting in the Way of My Smoking： Andy Huggins.wav":
None,

"20250818_The Secret to Winning Every Argument： Anjelah Johnson-Reyes.wav":
[(2,3),(7,10),(23,25),(29,31),(33,34),(46,48),(57,59)],

"20250821_Stop Asking My Age, Seriously... Masha Warfield.wav":
[(4,5),(8,10),(40,45),(58,60)],

"20250823_How Indian Wives Keep Husbands in Line： Zarna Garg.wav":
[(11,13),(28,29),(34,35),(40,42),(50,60)],

"20250909_How Getting Older Actually Works： Leighann Lord.wav":
[(12,15),(19,26),(29,31),(50,54),(58,59)],

"20250910_Why Waffle House Ain’t My Favorite Anymore： Lavell Crawford.wav":
[(25,27),(37,38),(58,59)],

"20250923_I Shouldn’t Be Single… But I Am： Preacher Lawson.wav":
[(9,11),(15,16),(22,24),(28,29),(48,50),(54,56),(59,60)],

"20251008_Only The Avengers Can Save Detroit： Ron Taylor.wav":
[(11,13),(15,16),(32,34),(38,39),(55,57),(59,60)],

"20251010_My Doctor Told Me I Have THIS…Ian Karmel.wav":
[(7,9),(21,30),(42,47),(53,55)],

"20251014_My Ex Was Actually Crazy： Preacher Lawson.wav":
[(6,7),(23,26),(39,41),(47,48)],

"20251015_Peanut Roasts the Whole Crowd： Jeff Dunham.wav":
[(2,5),(12,13),(16,17),(21,22),(41,44),(48,49)],

"20251024_Jeff Dunham Couldn’t Believe What Peanut Saw on Halloween.wav":
[(6,11),(25,28),(32,34),(36,39),(42,43),(47,55)],

"20251031_45 Minutes of Jeff Dunham Being ROASTED by Walter, Bubba J, Peanut & Achmed on Halloween 🎃.wav":
[(0,4),(19,21),(23,26),(29,30),(34,36),(41,42),(57,58)],

"20251031_Scarier Than Any Horror Movie (Bill Engvall, Lavell Crawford, Tom Papa & More).wav":
[(2,3),(24,26),(31,33),(52,53),(55,56)],

"20251102_I Never Thought My Life Would Turn Out Like This： Michael Yo.wav":
[(23,27),(33,35),(40,41),(50,53)],

"20251110_Having a Famous Granddad Isn’t What You Think： Samantha Hale.wav":
[(39,55)],

"20251112_I Failed College… Proudly： Bill Engvall.wav":
[(3,7)],

"20251125_Parenting in Brooklyn Hits Different： Sean Patton.wav":
[(9,13),(35,37),(39,41),(45,47)],

"20251130_38 Minutes of Jimmy O  Yang ｜ Good Deal.wav":
[(5,7),(10,12),(18,20),(24,26),(28,30),(38,43),(49,51),(54,55),(57,58)],

"20251218_WHY DO THEY MADE US BELIEVE EVERYONE WAS FAT？ ｜ Lace Larrabee.wav":
[(1,5),(11,13),(29,31),(35,37),(54,59)],

"20251231_46 Minutes of the BEST OF Kathleen Madigan.wav":
[(11,13),(17,18),(21,23)],

"20260117_Comedians on PARENTING： Why is it SO HARD？.wav":
[(10,11),(29,30),(49,50)],

"20260120_I was naked with the WRONG people in my 20s 😂 ｜ Lace Larrabee.wav":
[(18,20),(35,37),(46,48),(50,51)],

"20260121_DATING AT 60 ｜ With Late Bob Saget.wav":
[(6,8),(32,34),(50,52)]
}

'''Convert laughter segments in ground_truth (dict) into binary frames'''
def binary_frames(file, annotations, hop=512):
    y, sr = librosa.load(file, sr=None, duration=60)
    num_frames = int(np.ceil(len(y) / hop))
    # num_frames may not be the same length as real number of frames
    
    labels = np.zeros(num_frames)
    
    if annotations is None:
        return labels
    
    for start, end in annotations:
        start_frame = int(start * sr / hop)
        end_frame = int(end * sr / hop) + 1
        labels[start_frame:end_frame] = 1
    # binary: non-laughter = 0; laughter = 1
    return labels

'''Extract hyperparameters for machine learning'''
def extract_features(file, hop=512):
    y, sr = librosa.load(file, sr=None, duration=60)
    
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop)[0]
    
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
    
    delta_rms = np.diff(rms, prepend=rms[0])
    delta_centroid = np.diff(centroid, prepend=centroid[0])
    
    flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    
    # normalize
    rms = (rms_db - np.mean(rms_db)) / np.std(rms_db)
    zcr = (zcr - np.mean(zcr)) / np.std(zcr)
    centroid = (centroid - np.mean(centroid)) / np.std(centroid)
    delta_rms = (delta_rms - np.mean(delta_rms)) / np.std(delta_rms)
    delta_centroid = (delta_centroid - np.mean(delta_centroid)) / np.std(delta_centroid)
    flux = (flux - np.mean(flux)) / np.std(flux)
    
    return rms, zcr, centroid, delta_rms, delta_centroid, flux

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
    
    X_all = np.vstack(X_all)
    y_all = np.hstack(y_all)
    
    return X_all, y_all

files = list(ground_truth.keys())

'''Split videos into training and testing set'''

random.seed(42)
random.shuffle(files)

split_index = int(0.8 * len(files))
train_files = files[:split_index]
test_files = files[split_index:]

print("Train videos:", len(train_files))
print("Test videos:", len(test_files))

folder = "files_selected"

X_train, y_train = build_dataset(train_files, ground_truth, folder)
X_test, y_test = build_dataset(test_files, ground_truth, folder)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

'''Model--Logistic Regression, KNN, Decision Tree, Random Forest'''

param_grids = {

    "Logistic Regression": {
        "C": np.logspace(-3, 3, 10),
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    },

    "KNN": {
        "n_neighbors": list(range(1, 201, 5))
    },

    "Decision Tree": {
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },

    "Random Forest": {
        "n_estimators": [100],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    }
}
base_models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "KNN": KNeighborsClassifier(),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}
best_models = {}

print("\n===== HYPERPARAMETER TUNING (ROC-AUC) =====")

for name in base_models:

    grid = GridSearchCV(
        base_models[name],
        param_grids[name],
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    best_models[name] = grid.best_estimator_

    print(f"\n{name}")
    print("Best Parameters:", grid.best_params_)
    print("Best CV AUC:", round(grid.best_score_, 4))

plt.figure()
for name, model in best_models.items():

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n===== {name} TEST RESULTS =====")
    print("Best Parameters:", model.get_params())
    print("Test AUC:", round(roc_auc, 4))
    print("F1:", round(f1_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred), 4))
    print("Recall:", round(recall_score(y_test, y_pred), 4))
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))

    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()

plt.savefig("/Users/gloria/Desktop/ROC_Comparison_AllModels.png",
            dpi=300,
            bbox_inches="tight")
plt.close()