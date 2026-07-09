import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def _prefer_nvidia_on_windows(exe_path: str) -> bool:
    """Просит Windows рендерить exe_path на дискретном GPU.

    На Windows GPU для приложения выбирается не GLX-переменными (это Linux),
    а предпочтением 'High performance' в реестре
    HKCU\\Software\\Microsoft\\DirectX\\UserGpuPreferences - тот же механизм,
    что 'Параметры > Дисплей > Графика' и который учитывают драйверы NVIDIA/AMD
    в т.ч. для OpenGL. Пишем только под текущего пользователя (без админа),
    только для нашего интерпретатора, идемпотентно. True, если применено."""
    try:
        import winreg  # только Windows; на других ОС сюда не попадаем
    except ImportError:
        return False
    key_path = r"Software\Microsoft\DirectX\UserGpuPreferences"
    desired = "GpuPreference=2;"  # 2 = High performance (дискретная карта)
    try:
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            try:
                current, _ = winreg.QueryValueEx(key, exe_path)
            except FileNotFoundError:
                current = None
            if current != desired:
                winreg.SetValueEx(key, exe_path, 0, winreg.REG_SZ, desired)
        return True
    except OSError:
        return False


# Настройки рендера по умолчанию, чтобы не задавать их вручную каждый запуск.
# Оба дочерних процесса (ML и UI) наследуют os.environ, поэтому ставим до Popen.
_system = platform.system()

# Сглаживание сцены: none обходит падение шейдера vtkSSAAPass на некоторых
# GPU-контекстах. setdefault => можно переопределить из шелла (OIL_AA=fxaa ...).
os.environ.setdefault("OIL_AA", "none")

# Заставляем рисовать 3D на дискретной NVIDIA. Способ зависит от ОС:
_gpu_mode = "по умолчанию (встроенный GPU / выбор ОС)"
if _system == "Linux" and os.path.exists("/proc/driver/nvidia"):
    # Linux: PRIME render offload через GLX-переменные (проприетарный драйвер
    # реально загружен - иначе не трогаем, чтобы не сломать GLX без NVIDIA).
    os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    _gpu_mode = "NVIDIA (PRIME offload)"
elif _system == "Windows" and shutil.which("nvidia-smi"):
    # Windows: nvidia-smi ставится вместе с драйвером NVIDIA -> дискретка есть.
    # Плоские точки для частиц газа (шейдер точек-сфер на части Windows/GPU
    # контекстов не компилируется, как SSAA). Имя переменной должно совпадать
    # с тем, что читает RenderProfile: OIL_POINT_SPHERES (без префикса __).
    os.environ.setdefault("OIL_POINT_SPHERES", "0")
    # Прописываем предпочтение High performance для текущего python.exe.
    if _prefer_nvidia_on_windows(sys.executable):
        _gpu_mode = "NVIDIA (High performance в реестре Windows)"

print(f"Рендер сцены: GPU={_gpu_mode}, OIL_AA={os.environ['OIL_AA']}")

# Запуск ML-сервера
ml = subprocess.Popen([
    sys.executable, "scripts/run_demo.py",
    "--event-server", "--headless",
    "--cursor-on", "--real-mouse",
    "--publish-camera",
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