# Muse EEG Emotion Recognition

This folder contains two clear parts:

- `Algorithm`: prepares SEED-IV data, extracts features, trains models, and
  saves the best model.
- `Service`: reads live Muse EEG data and shows emotion predictions in a small
  browser interface.

## Folder structure

```text
Backend_emotion_recognition/
|-- Algorithm/
|   |-- config.py
|   |-- dataset_preparation.py
|   |-- feature_extraction.py
|   |-- train_model.py
|   |-- inference.py
|   |-- processed_data/
|   |-- features/
|   `-- saved_model/
|       `-- trained_model.pkl
|-- Service/
|   |-- app.py
|   |-- detector.py
|   |-- muse_client.py
|   |-- main.py
|   `-- static/
|-- requirements.txt
`-- README.md
```

## Electrode mapping

The model uses only four SEED-IV channels. They are the closest available
positions to the four Muse electrodes:

| Muse electrode | SEED-IV channel used |
|---|---|
| TP9 | TP7 |
| TP10 | TP8 |
| AF7 | AF3 |
| AF8 | AF4 |

The channel order is kept identical during training and live detection.

## Installation

Open PowerShell in `D:\BOOK\eeg_project`, activate the environment, and run:

```powershell
eegp\Scripts\activate
cd Backend_emotion_recognition
pip install -r requirements.txt
```

Random Forest, Extra Trees, and SVM work with the required packages. XGBoost
and LightGBM are optional:

```powershell
pip install xgboost lightgbm
```

## Algorithm commands

Run these commands from the `Backend_emotion_recognition` folder and in this
order:

```powershell
python -m Algorithm.dataset_preparation
python -m Algorithm.feature_extraction
python -m Algorithm.train_model --skip-svm --skip-optional-boosters
```

The commands create:

- `Algorithm/processed_data/processed_windows.npz`
- `Algorithm/features/features.npz`
- `Algorithm/saved_model/trained_model.pkl`

The last command trains the two simple tree models and saves the better one.
To compare every installed model, run:

```powershell
python -m Algorithm.train_model
```

## Start live Muse detection

The default Muse BLE address is set near the top of `Service/muse_client.py`.
Start the service with:

```powershell
python Service\main.py
```

Then open `http://127.0.0.1:5000` in a browser. Use the buttons in this order:

The FastAPI interactive documentation is available at
`http://127.0.0.1:5000/docs`.

1. Connect Device
2. Start Detection
3. Stop Detection when finished
4. Disconnect

Detection starts after each Muse channel contains a complete 4-second window.
Muse data is resampled from 256 Hz to the model's 200 Hz sampling rate before
the same preprocessing and feature extraction used during training.

## Change the dataset location

The dataset path is intentionally simple and fixed. If the dataset is moved,
edit only `RAW_EEG_DIR` in `Algorithm/config.py`.
