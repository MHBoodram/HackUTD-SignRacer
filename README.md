# SignRacer

SignRacer is a Python game where you race by signing letters in front of your
webcam. It includes a solo Practice mode and an online multiplayer race mode.
The app uses OpenCV and MediaPipe to read hand movements and a scikit-learn
classifier to predict signs.

## Requirements

- Windows (the commands below use PowerShell)
- Python 3.12 recommended
- A webcam for sign input
- An internet connection for multiplayer

## Set up

Clone the repository and enter its folder:

```powershell
git clone https://github.com/MHBoodram/HackUTD-SignRacer.git
cd HackUTD-SignRacer
```

Create a virtual environment, activate it, and install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell does not allow environment activation, run commands with the
virtual environment's interpreter directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Always use the same environment to install dependencies and start the game.
This helps avoid errors such as `ModuleNotFoundError: No module named 'pygame'`
or using a different NumPy installation from Anaconda `base`.

## Start the game

```powershell
python ui.py
```

The app loads the classifier from `models/asl_classifier.plk` on startup. A
model is included in the repository so you can launch the game without first
training one. The checked-in model is limited: it recognizes only a few
classes. For better recognition, prepare a full image dataset and train your
own model as described below.

## Practice

Select **Practice** on the main menu. The screen shows a target sign and a
camera preview.

- Sign the displayed letter in view of the camera to score.
- Press **Enter** to skip the current target.
- Press **Tab** to cycle between Letters, Numbers, and All.
- Press **Esc** to return to the menu.
- Use **Change camera** if the wrong camera is selected.
- Toggle **Show hints** to show or hide example images.

On Windows, allow desktop apps to access the camera in **Settings → Privacy &
security → Camera**. Close other camera applications if the preview is missing
or frozen.

## Train the model with your images

The full training image dataset is not included in this GitHub repository.
Provide your own images under `data/asl_dataset`, with one single-character
folder per sign. The generator accepts `.jpg`, `.jpeg`, `.png`, `.bmp`, and
`.webp` images. For example:

```text
data/
└── asl_dataset/
    ├── a/
    │   ├── image_001.jpg
    │   └── image_002.jpg
    ├── b/
    │   ├── image_001.jpg
    │   └── image_002.jpg
    └── ...
```

Use multiple images for each class. The trainer requires enough samples to
create a stratified training and test split; it will refuse to save a model if
the dataset is too small. The generated feature CSV is written to
`data/asl_dataset.csv`.

Generate features and train the classifier from the repository root:

```powershell
python -m src.vision.dataset_generator
python -m src.vision.model_trainer
```

The generator includes letter folders by default and skips digit folders. To
train number signs too, set `INCLUDE_DIGITS = True` near the top of
`src/vision/dataset_generator.py` before running the generator. Training
replaces `models/asl_classifier.plk`; restart the game to load the new model.

## Multiplayer

Multiplayer uses the project's configured SpacetimeDB service and requires an
internet connection.

1. Select **Create Room** and share the displayed room code with your opponent.
2. The opponent selects **Join with Code**, enters the code, and presses
   **Enter**.
3. Both players allow camera access and get ready; the race starts after both
   are ready.
4. Sign the prompted letters to advance. Press **Esc** to leave the room.

To run two clients on the same computer for testing, open a second terminal and
start another profile:

```powershell
python ui.py 2
```

The default profile is `1`. Each profile uses its own local SpacetimeDB
identity. Keep generated `stdb_token_*.json` files private; do not share or
publish them.

## Troubleshooting

- **A Python module is missing:** activate `.venv`, or use
  `.\.venv\Scripts\python.exe` for both installation and launch.
- **Camera preview is missing:** close other camera apps, check Windows camera
  privacy settings, and try **Change camera** in Practice.
- **The model file is missing:** generate features and train a model using the
  commands above.
- **Training says there are too few samples:** add more images, especially for
  labels with fewer examples, then regenerate the CSV and train again.
- **Predictions are inaccurate:** use clear, varied images for every sign and
  make sure the webcam shows the signing hand clearly.
