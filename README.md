# pushup-professor

## Project Overview
Python script that uses YOLO model to detect body positions to count push up reps. When the inital video stream was laggy, I decided to look more into why and made improvements to the code. In the end, I ran different experiments to obtain quantitative data on those improvements.

### Improvements
* Stable angle measurements: while not a performance improvement, this resulted in more accurate rep counting. Instead of using angles calculated from just the current frame, a rolling average is used. More specifically, the average angle value of the last 5 frames.
* Image size: ability to specify image size passed into the model. Smaller sizes may result in greater FPS, at the expense of less accurate inference, and thus less accurate rep counting.
* Inference skipping: ability to run inference every n frames, instead of on every frame. Again, may result in less accurate rep counting.

## Demo
<video src="./final.mp4" controls loop width="100%"></video>

## Setup
Set up a Python venv and run `pip install -r requirements.txt`

## How to Run
Run `py main.py` with the following options:
* `--imgsz` --- sets the image size passed into the model for inference (default 640).
* `--skip-frames` --- how many frames to skip in-between running inference. For example, `--skip-frames 0` runs inference on every frame, while `--skip-frames 1` runs inference every other frame (default 0).
* `--log` --- toggles on logging capabilities.

## Performance Results

## Discussion of Bottlenecks