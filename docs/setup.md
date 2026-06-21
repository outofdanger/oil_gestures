# Setup

## Python runtime

Use the standard GIL-enabled CPython 3.14.x runtime. The tested and locally
pinned version is CPython 3.14.6; `.python-version` allows compatible runtime
managers such as pyenv to select it automatically. The free-threaded `3.14t`
build has not been validated with MediaPipe and is not supported yet.

Create an isolated environment and install dependencies on macOS or Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python --version
```

The final command must report Python 3.14.x. On Windows, use:

```powershell
py -3.14 -m venv .venv
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

Real cursor control uses the X11 XTest extension through `python-xlib`, which is
installed from `requirements.txt`. Run a backend-only check before opening the
camera:

```bash
python app/main.py --mouse-diagnostics --real-mouse
python app/main.py --test-mouse-move --real-mouse
```

The first command should report `backend: x11-xtest`, `display_server: x11`,
the current pointer position, and desktop bounds.

Native Wayland deliberately restricts global synthetic input. Gesture
recognition and dry-run work normally there, but `--real-mouse` exits with a
clear explanation. Select an X11/Xorg desktop session at login when real cursor
control is required. A missing `DISPLAY` usually means the app was launched
outside the graphical session (for example, over a plain SSH connection).
