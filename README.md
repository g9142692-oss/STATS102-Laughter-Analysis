# STATS102 Final Project — Laughter Analysis

Final project for **STATS102** (Spring 2026), analyzing laughter in stand-up comedy recordings using audio-derived features and statistical / machine-learning models.

## Overview

The project models laughter in short comedy clips. Audio features are extracted per video and used to characterize and predict laughter:

| Feature | Description |
| --- | --- |
| `laughter_duration` | Duration of laughter in the clip |
| `laughter_volume` | Loudness of laughter |
| `onset_rate` | Rate of laughter onset |
| `expressiveness` | Expressiveness measure |
| `pitch_variability` | Variability in pitch |
| `energy_variability` | Variability in signal energy |

## Repository contents

- `Zihan Zhang_Final report.docx` — final written report
- `Zihan Zhang_Final Presentation.pptx` — final presentation slides
- `laughter_analysis_results.csv` — extracted audio features per video (`video_id`, six feature columns)
- `Link of recordings (.wav form).docx` — links to the source `.wav` recordings
- `ground_truth_dict.html` — ground-truth dictionary of laughter segments
- `OLS.html`, `Lasso.html`, `PCA.html` — regression / dimensionality-reduction analyses
- `Model selection (logistic, KNN, decision tree, random forest).html` — model selection across classifiers
- `Random Forest_extract.html`, `Random_selection (30 videos).html` — random forest and sampling details

## Methods

- **Regression**: OLS, Lasso
- **Dimensionality reduction**: PCA
- **Classification**: logistic regression, KNN, decision tree, random forest

## Author

Zihan Zhang
