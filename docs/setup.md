# Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run gesture recognition. The secondary cursor subsystem starts disabled:

```bash
python3 scripts/run_demo.py
```

Start with cursor control on, still in safe dry-run mode:

```bash
python3 scripts/run_demo.py --cursor-on
```

Run with real cursor control only when you intentionally want OS mouse events:

```bash
python3 scripts/run_demo.py --cursor-on --real-mouse
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
python3 app/main.py --mouse-diagnostics --real-mouse
python3 app/main.py --test-mouse-move --real-mouse
```

The first command should report `backend: x11-xtest`, `display_server: x11`,
the current pointer position, and desktop bounds.

Native Wayland deliberately restricts global synthetic input. Gesture
recognition and dry-run work normally there, but `--real-mouse` exits with a
clear explanation. Select an X11/Xorg desktop session at login when real cursor
control is required. A missing `DISPLAY` usually means the app was launched
outside the graphical session (for example, over a plain SSH connection).
