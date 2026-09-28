# Android integration plan (playbook prompt H)

The navigation core does not change for Android. The app's only job is to turn
Android callbacks into the same `SensorSample` stream that `ReplaySource`
produces from IO-VNBD, then display the engine's output.

## Sensor mapping

| Android API | `SensorSample` field | Notes |
|---|---|---|
| `Sensor.TYPE_ACCELEROMETER` (m/s², phone frame, includes gravity) | `acc` (3,) | Register at `SENSOR_DELAY_GAME` (≈50 Hz) or faster. Do **not** use `TYPE_LINEAR_ACCELERATION` or an Earth-frame rotation. The engine levels and aligns the raw body-frame vector itself. |
| `Sensor.TYPE_GYROSCOPE` (rad/s, phone frame) | `gyro` (3,) | Use the calibrated gyro, not `_UNCALIBRATED`. The EKF estimates the remaining bias. |
| `SensorEvent.timestamp` (ns, elapsedRealtimeNanos) | `t` (s) | One monotonic clock for IMU and GNSS. |
| `LocationListener.onLocationChanged` / `FusedLocationProvider` | `gnss = GnssFix(x, y, speed, yaw, pos_std)` | Convert lat/lon to the session's local ENU origin with `data_io.latlon_to_local`. `yaw = heading_to_yaw(location.bearing)`. `pos_std = location.accuracy`. |
| `GnssStatus` / no fix for `gnss_timeout_s` | `gnss = None` | "GNSS healthy" means a fix with `accuracy < 20 m` and `hasBearing()` when moving. Everything else is `None`. |

IO-VNBD difference: the IO-VNBD logger stored accelerometer X/Y already
rotated by the phone's azimuth. `data_io.standardise` undoes that, so live
Android data and the replayed dataset reach the engine in the same phone-body
frame.

## Runtime layout

```
SensorManager / LocationManager callbacks
        │   (merge by timestamp, 10–200 Hz)
        ▼
SensorSample queue ──► NavigationEngine.step()   ← same Python code, via Chaquopy,
        │                                           or a Kotlin port of engine.py / ekf2d.py
        ▼
UI: mode banner (GNSS+INS / DEAD RECKONING), disp_x/disp_y marker, sigma circle
```

* **Calibration.** At session start, collect up to `preprocess.calib_max_s` of samples with healthy GNSS and call `fit_alignment`. Re-fit if the phone is remounted: a jump in the gravity direction or `alignment.fit_corr < 0.5`.
* **MotionNet.** Export with `torch.onnx.export(model.net, torch.zeros(1, window, 5), "motionnet.onnx")` and run it with ONNX Runtime Mobile. It has 18k parameters and takes under 1 ms per call. Feed it the same 10 Hz block-averaged features that `engine._feed_model` builds.
* **Map.** Ship an offline OSM extract (`src.constraints.fetch_osm`) and draw the map with MapLibre Native.
* **Display.** Draw `disp_x`/`disp_y`, not `x`/`y`, so the marker eases onto GNSS after a tunnel instead of teleporting.

## What must not change

Android UI code must not add inputs to the engine. Wheel speed, OBD and CAN
stay out, and GNSS never enters the engine while it is flagged unhealthy.
The unit tests in `tests/test_no_gnss_leakage.py` check this at the engine
boundary.
