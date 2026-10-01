import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


df = pd.read_csv('laughter_analysis_results.csv')
df = df.drop(columns=['video_id'])
corr_matrix = df.corr(method="spearman")

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Spearman Correlation Matrix")
plt.legend()

plt.savefig("/Users/gloria/Desktop/Correlation Matrix.png",
            dpi=300,
            bbox_inches="tight")
plt.close()

speaker_features = ["onset_rate", "expressiveness",
                    "pitch_variability", "energy_variability"]

X = df[speaker_features]
y_duration = df['laughter_duration']
y_volume = df["laughter_volume"]

# Lasso & cross-validation
lasso_volume = Pipeline([
    ("scaler", StandardScaler()),   
    ("lasso", LassoCV(cv=5, random_state=42))
])

lasso_volume.fit(X, y_volume)

lasso_duration = Pipeline([
    ("scaler", StandardScaler()),   
    ("lasso", LassoCV(cv=5, random_state=42))
])

lasso_duration.fit(X, y_duration)

lasso = lasso_duration.named_steps["lasso"]

print("Best alpha_duration:", lasso.alpha_)
print("R²:", lasso_duration.score(X, y_duration))

coefficients = pd.Series(
    lasso.coef_,
    index=speaker_features
)

print("\nLasso Coefficients_Duration:")
print(coefficients)

lasso = lasso_volume.named_steps["lasso"]

print("Best alpha_volume:", lasso.alpha_)
print("R²:", lasso_volume.score(X, y_volume))

coefficients = pd.Series(
    lasso.coef_,
    index=speaker_features
)

print("\nLasso Coefficients_Volume:")
print(coefficients)