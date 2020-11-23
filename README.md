# pytrdrop
Python implementation of the trdrop application

The aim is to replicate the features of [trdrop](https://github.com/cirquit/trdrop),
but using Python with OpenCV.

# Installation & Requirements
Requires Python 3.
Install the dependencies by running the following command:
```
pip install -r requirements.txt
```
or:
```
python -m pip install -r requirements.txt
```

If you use the `pipenv` module, you can create a virtual environment with the
following command:
```
pipenv update
```
or:
```
python -m pipenv update
```

# Usage
You can use the help command to get a list of available options with the
following command:
```
python fps_2_chart.py -h
```

Here is what that output looks like:
```
usage: video_analyser.py [-h] [-o OUTPUT] [-t THRESHOLD] INPUT

Analyze framerate, frame drops, and frame tears of a video file.

positional arguments:
  INPUT                 Video File

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output filename (Default will be named after input file).
  -t THRESHOLD, --threshold THRESHOLD
                        Pixel difference threshold to count as duplicate frames, must be an integer between 0 and 255.
                        A value of 0 will count all all frames as unique, while 255 will only count
                        frames that are 100 percent different (Default is 5).
```
