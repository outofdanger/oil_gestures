# Setup

## Python runtime

Use the standard GIL-enabled CPython 3.12.x runtime. The tested and locally
pinned version is CPython 3.12.3; `.python-version` allows compatible runtime
managers such as pyenv to select it automatically.

Create an isolated environment and install dependencies on macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
```

The final command must report Python 3.12.x. On Windows, use:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
```

The binary dependency set is available for these primary targets:

- macOS 13 or newer on Apple Silicon
- Windows x86-64
- Linux x86-64 with glibc 2.28 or newer

MediaPipe/OpenCV do not currently provide the complete wheel set for macOS
Intel, Linux ARM64, or Windows ARM64. Building those native dependencies from
source is outside the supported setup.

## Run

Run gesture recognition. The secondary cursor subsystem starts disabled:

```bash
python scripts/run_demo.py
```

Start with cursor control on, still in safe dry-run mode:

```bash
python scripts/run_demo.py --cursor-on
```

Run with real cursor control only when you intentionally want OS mouse events:

```bash
python scripts/run_demo.py --cursor-on --real-mouse
```

On macOS, grant permissions in System Settings:

- Camera permission for VS Code or Terminal
- Accessibility permission for VS Code or Terminal

If the cursor does not move but the debug window shows landmarks, the missing
permission is usually Accessibility.

## Linux

The camera path prefers the V4L2 backend. The user running Python must be able
to read the selected `/dev/video*` device; check this with:

```bash
ls -l /dev/video*
groups
```

The default Linux camera negotiation requests MJPG at 1280x720@30. MediaPipe
processes a smaller 640x360 frame; the OpenCV preview keeps the camera's native
frame. At startup, inspect the `Camera opened` log line. It should normally
report `backend=V4L2`, `FOURCC=MJPG`, and at least 15 FPS.

If the driver still reports less than 15 FPS, list its supported modes:

```bash
v4l2-ctl --device /dev/video0 --list-formats-ext
```

Then select a supported MJPG mode in `configs/default.yaml`. The conservative
fallback is `width: 640`, `height: 480`, `fps: 30`, with
`preferred_fourcc: "MJPG"`. Low light can also make some webcams increase
exposure time and reduce their real frame rate even when 30 FPS was requested.

Real cursor control uses one of two backends depending on the session type:

- **X11/Xorg**: the X11 XTest extension through `python-xlib` (installed from
  `requirements.txt`).
- **Native Wayland**: a virtual absolute pointer created through
  `/dev/uinput`. XTest only affects XWayland clients, not the compositor, so
  it cannot move the real cursor under Wayland — uinput events go through the
  kernel evdev/libinput stack instead, the same path a physical touchscreen
  uses, which the compositor does see.

Run a backend-only check before opening the camera:

```bash
python app/main.py --mouse-diagnostics --real-mouse
python app/main.py --test-mouse-move --real-mouse
```

The first command reports `backend: x11-xtest` and `display_server: x11` on
an X11/Xorg session, or `backend: uinput` and `display_server: wayland` on a
native Wayland session, plus the current pointer position and desktop bounds.

### Wayland: uinput permissions

The uinput backend needs read/write access to `/dev/uinput`:

```bash
sudo modprobe uinput
ls -l /dev/uinput
```

Grant access either by joining the `input` group:

```bash
sudo usermod -aG input $USER
# log out and back in for the new group membership to take effect
```

or with a persistent udev rule:

```bash
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/99-uinput.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Without one of these, `--real-mouse` fails fast with a specific error
explaining which permission is missing, instead of silently falling back to
dry-run. A missing `DISPLAY` (on X11/XWayland) usually means the app was
launched outside the graphical session (for example, over a plain SSH
connection); this does not affect the uinput path, which does not need
`DISPLAY` at all.
