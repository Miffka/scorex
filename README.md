# SCORe EXtractor

Tennis score extractor.

# How to use

## 1. Installation

```
pip install -r requirements.txt
pip install -e .
```

## 2. Run in demo mode with visualization

```
python scorex/model/predictor.py data/raw/top-100-shots-rallies-2018-atp-season.mp4 --visualize
```

# Logic behind

1. Find bounding box of score tableu in bottom left corner with OpenCV.
2. Parse it with Tesseract.
