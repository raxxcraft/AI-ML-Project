# Driver Drowsiness Detection (Corrected)

## Features
- Real-time webcam drowsiness monitoring.
- Alarm sound for sustained eye closure.
- Time-based closed-eye scoring (`Closed Time`) for FPS-stable behavior.
- Uses CNN model (`dl_model/drowsiness_cnn.h5`) if available.
- Runs in conservative fallback mode when model file is missing.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```powershell
python app.py
```

## Controls
- Press `q` to quit.

## Optional arguments
```powershell
python app.py --camera 0 --alert-seconds 3.0 --decay-rate 1.0 --fallback-confirm-frames 8 --fallback-eye-ar-threshold 0.22 --cooldown 1.5
```

## Notes
- Best accuracy requires `dl_model/drowsiness_cnn.h5`.
- Alert default is about 3 seconds of continuous closed-eye time.
- If model is missing, fallback uses repeated evidence plus eye-box shape.
- In fallback mode, status labels use `*` (for example `Verifying*`, `Closed*`).
