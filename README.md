# RockLook

RockLook plays rock music when your head tilts downward and pauses the music when you look up again.

## Tech stack

- Python
- OpenCV
- MediaPipe FaceMesh
- pygame mixer

## What it ships

- Webcam feed runs
- Music starts when you look down
- Music pauses/stops when you look up
- Threshold value shown on screen
- Easy calibration step at startup

## Files

- `rocklook.py` — main program
- `rock.mp3` — your music file

## Setup

1. Install Python 3.10 or newer.
2. Install dependencies:

   ```bash
   pip install opencv-python mediapipe pygame
   ```

3. Put an `.mp3` file in the same folder as the script, or rename it to `rock.mp3`.

## Run

```bash
python rocklook.py
```

If your webcam does not open, try:

```bash
python rocklook.py --camera 1
```

If your music file has a different name:

```bash
python rocklook.py --music my_song.mp3
```

## How it works

The program uses FaceMesh landmarks to estimate head pitch. It computes a simple downward score from the position of the nose relative to the eyes. During startup it calibrates your neutral position for a few frames, then uses that baseline to decide whether you are looking down.

A small hysteresis gap is used so the music does not flicker on and off too quickly.

## Troubleshooting

- **Webcam not found:** try `--camera 1`
- **MediaPipe import error:** use Python 3.10+
- **No sound:** make sure your `.mp3` file exists and your system volume is on
- **Music keeps toggling:** stand still during calibration and face the camera straight on

## Notes

This is a simple demo project, so the head-tilt detection is heuristic rather than perfect eye-gaze tracking.
