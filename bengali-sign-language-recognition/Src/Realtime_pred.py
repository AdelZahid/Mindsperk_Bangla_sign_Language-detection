"""
realtime_predict.py  —  BaSL Glove real-time recognition (single script, no CSV)
================================================================================

Connect glove -> press button -> capture 20x11 -> scale (as in training) ->
XGBoost predict -> print word + confidence + top-3 -> loop. No files, no speech.

Run:
    pip install pyserial joblib xgboost scikit-learn numpy --break-system-packages
    python realtime_predict.py --port COM6

Needs (saved once after newprj.py):
    basl_model.joblib, basl_scaler.joblib, basl_labels.joblib
        import joblib
        joblib.dump(best_model,'basl_model.joblib')
        joblib.dump(scaler,'basl_scaler.joblib')
        joblib.dump(label_encoder,'basl_labels.joblib')

Firmware: your existing Fixed_gyro data-collection sketch works as-is. This
script also accepts the "RAW,timestep,..." format, so either firmware is fine
(see PARSING below).

--------------------------------------------------------------------------------
TWO THINGS YOUR SPEC DOC HAD SUBTLY WRONG — this script does the correct thing:

1. PREPROCESS ORDER. The spec said "flatten to (220,) THEN StandardScaler".
   Your training (newprj.py) does the opposite: it scales the (20,11) array
   PER-CHANNEL first, THEN flattens. Your saved scaler has 11 features, not
   220, so it literally cannot accept a flattened vector. Correct order, which
   this script uses:  scale (20,11) -> flatten -> (1,220).

2. LINE FORMAT. The spec's "RAW,timestep,<11>" (13 fields) is NOT what the
   data-collection firmware emits ("sample_id,label,timestep,<11>", 14 fields).
   This script auto-detects BOTH, so it doesn't matter which you flash.
--------------------------------------------------------------------------------
"""

import argparse
import time
import numpy as np
import joblib

N_CHANNELS = 11
N_TIMESTEPS = 20
PARTIAL_TIMEOUT_S = 1.5    # drop an incomplete capture if it stalls


def parse_row(line):
    """Return (timestep, [11 floats]) for either firmware format, else None."""
    p = [x.strip() for x in line.split(",")]
    # Format A:  RAW,timestep,f1..gz          (13 fields)
    if p and p[0].upper() == "RAW" and len(p) == 2 + N_CHANNELS:
        try:
            return int(p[1]), [float(x) for x in p[2:2 + N_CHANNELS]]
        except ValueError:
            return None
    # Format B:  sample_id,label,timestep,f1..gz   (14 fields)
    if len(p) == 3 + N_CHANNELS:
        try:
            return int(p[2]), [float(x) for x in p[3:3 + N_CHANNELS]]
        except ValueError:
            return None
    return None


def predict(buf, model, scaler, label_encoder):
    arr = np.asarray(buf, dtype=float)                 # (20, 11)
    scaled = scaler.transform(arr)                     # per-channel, 11-feature scaler
    flat = scaled.reshape(1, -1)                       # (1, 220), C-order = t*11 + c
    proba = model.predict_proba(flat)[0]
    order = np.argsort(proba)[::-1]
    classes = label_encoder.classes_
    return [(classes[i], float(proba[i])) for i in order[:3]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Serial port e.g. COM6 or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--model", default="basl_model.joblib")
    ap.add_argument("--scaler", default="basl_scaler.joblib")
    ap.add_argument("--labels", default="basl_labels.joblib")
    args = ap.parse_args()

    import serial
    from serial import SerialException

    model = joblib.load(args.model)
    scaler = joblib.load(args.scaler)
    label_encoder = joblib.load(args.labels)
    print(f"Loaded model. {len(label_encoder.classes_)} classes: {list(label_encoder.classes_)}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except SerialException as e:
        print(f"Could not open {args.port}: {e}")
        return

    print(f"Connected to {args.port}. Press the glove button to sign. Ctrl+C to stop.")
    print("-" * 60)

    buf = []
    count = 0
    last_row_time = None

    try:
        while True:
            try:
                raw = ser.readline()
            except SerialException:
                print("\n[serial disconnected] retrying in 2s...")
                time.sleep(2)
                try:
                    ser.close(); ser.open()
                except Exception:
                    pass
                continue

            # drop a stalled partial capture
            if buf and last_row_time and (time.time() - last_row_time) > PARTIAL_TIMEOUT_S:
                print(f"\n  [timeout] discarded incomplete capture ({len(buf)}/20)")
                buf = []

            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue

            if line.startswith("CAPTURING"):
                buf = []
                continue
            if line == "SAMPLE_DONE":
                continue   # boundary handled by timestep/length below

            parsed = parse_row(line)
            if parsed is None:
                # status text from firmware (MPU ok, READY, etc.)
                print(f"[esp32] {line}")
                continue

            ts, feats = parsed
            if ts == 0:
                buf = []            # a new sign starts at timestep 0
            buf.append(feats)
            last_row_time = time.time()
            print(f"\r  capturing {len(buf):2d}/20 " + "#" * len(buf), end="", flush=True)

            if len(buf) == N_TIMESTEPS:
                top = predict(buf, model, scaler, label_encoder)
                buf = []
                count += 1
                word, conf = top[0]
                print(f"\r  PREDICT -> {word.upper():<10} ({conf*100:5.1f}%)   [#{count}]")
                alts = "  |  ".join(f"{w} {p*100:.0f}%" for w, p in top)
                print(f"     {alts}\n")

    except KeyboardInterrupt:
        print(f"\nStopped. {count} predictions made.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()