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
python video_analyzer.py -h
```

Here is what that output looks like:
```
usage: video_analyser.py [-h] [-o OUTPUT] [-t THRESHOLD] [-s {0,1,2,3}] INPUT

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
                        frames that are 100 percent different (Default: 5).
  -s {0,1,2,3}, --save {0,1,2,3}
                        Save the video frames of the original video as well as a version with duplicated frames removed.
                        A value of 0 will not save any video files.
                        A value of 1 will only save the original video file.
                        A value of 2 will only save the version with duplicated frames removed.
                        A value of 3 will save both the original and version with duplicated frames removed.
                        Note that saving the video file(s) can drastically increase the program's runtime. (Default: 0)
```
