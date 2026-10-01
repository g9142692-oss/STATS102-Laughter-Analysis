import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr

# Regression 1--Predict laughter duration
df = pd.read_csv('laughter_analysis_results.csv')
df = df.drop(columns=['video_id'])

speaker_features = ["onset_rate", "expressiveness",
                    "pitch_variability", "energy_variability"]


Y_duration = df["laughter_duration"]
X_duration = sm.add_constant(df[speaker_features])

model_duration = sm.OLS(Y_duration, X_duration).fit()

print("\n===== Regression: Laughter Duration =====")
print(model_duration.summary())

# Regression 2--Predict laughter duration
Y_volume = df["laughter_volume"]
X_volume = sm.add_constant(df[speaker_features])

model_volume = sm.OLS(Y_volume, X_volume).fit()

print("\n===== Regression: Laughter Volume =====")
print(model_volume.summary())