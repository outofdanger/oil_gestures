import subprocess
import sys
import os
import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# Запуск ML-сервера
ml = subprocess.Popen([
    sys.executable, "scripts/run_demo.py",
    "--event-server", "--headless",
    "--cursor-on", "--real-mouse"
])

# Запуск UI
env = os.environ.copy()
env["PYTHONPATH"] = ROOT
ui = subprocess.Popen([sys.executable, "app/ui_main.py"], env=env)

print("ML-сервер и UI запущены. Закройте окно приложения для выхода.")

# Ждём завершения UI
ui.wait()

# Завершаем ML
ml.terminate()
ml.wait()
print("Завершено.")