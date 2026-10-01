import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm

df = pd.read_csv('laughter_analysis_results.csv')
df = df.drop(columns=['video_id'])
speaker_features = ["onset_rate", 
                    "expressiveness",
                    "pitch_variability", 
                    "energy_variability"]

X = df[speaker_features]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''PCA'''
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

# Component loadings
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(len(speaker_features))],
    index=speaker_features
)

print("\nPCA Loadings:")
print(loadings)

df["PC1"] = X_pca[:, 0]
df['PC2'] = X_pca[:, 1]

Y_duration = df["laughter_duration"]
Y_volume = df['laughter_volume']

# OLS regression: Duration & PC1
X_reg = sm.add_constant(df["PC1"])

model_duration = sm.OLS(Y_duration, X_reg).fit()

print("\n===== Regression: Duration ~ PC1 =====")
print(model_duration.summary())

# OLS regression: Volume & PC1
X_reg = sm.add_constant(df["PC1"])

model_volume1 = sm.OLS(Y_volume, X_reg).fit()

print("\n===== Regression: Volume ~ PC1 =====")
print(model_duration.volume1())

# OLS regression: Volume & PC2
X_reg = sm.add_constant(df["PC2"])

model_volume2 = sm.OLS(Y_volume, X_reg).fit()

print("\n===== Regression: Volume ~ PC2 =====")
print(model_duration.volume2())

plt.figure()
plt.scatter(df["PC1"], df["laughter_duration"])
plt.xlabel("PC1")
plt.ylabel("Laughter Duration")
plt.savefig("/Users/gloria/Desktop/PC1 vs Laughter Duration.png",
            dpi=300,
            bbox_inches="tight")

plt.figure()
plt.scatter(df["PC2"], df["laughter_volume"])
plt.xlabel("PC2")
plt.ylabel("Laughter Volume")
plt.savefig("/Users/gloria/Desktop/PC1 vs Laughter Volume.png",
            dpi=300,
            bbox_inches="tight")