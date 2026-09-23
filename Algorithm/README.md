# Algorithm commands

Open PowerShell in the `Backend_emotion_recognition` folder, then run:

```powershell
python -m Algorithm.dataset_preparation
python -m Algorithm.feature_extraction
python -m Algorithm.train_model --skip-svm --skip-optional-boosters
```

The first command creates EEG windows, the second creates 144 features for each
window, and the third trains Random Forest and Extra Trees and saves the better
model in `Algorithm/saved_model/trained_model.pkl`.

To compare SVM and any separately installed XGBoost or LightGBM models too:

```powershell
python -m Algorithm.train_model
```
