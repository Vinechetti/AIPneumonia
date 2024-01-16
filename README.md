# AIPneumonia
AI Pneumonia detector, created as a project for university.

# Usage
Remember to install the dependencies using:
```
pip install -r requirements.txt
```
## GUI
Usage is pretty simple and straightforward. Run:
```
python gui.py
```
You will be asked if you want to recalculate the model accuracy (the one calculated beforehand is ~84%).  The repository also contains images for calculating the accuracy, which you can also use as examples for testing the app.

## CLI
Usage:
```
python cli.py <image_path>
```

## Additional notes:
The program can return false positives or negatives.

The model was trained using the following dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/versions/2

