# Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run gesture recognition. Cursor control starts off and can be toggled with the
configured gesture, default `MIDDLE_PINCH`:

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
