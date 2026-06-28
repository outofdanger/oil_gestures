"""Кросс-платформенный профиль рендера.

Единственное место в коде, где сосредоточены различия между macOS / Linux /
Windows и особенности железа (Retina, software-рендер на серверах без GPU,
HiDPI на Windows/Wayland). Везде ещё `platform.system()` встречаться не должен —
вместо этого код берёт готовые параметры из :class:`RenderProfile`.

Любой параметр можно переопределить переменной окружения, чтобы тюнить под
конкретную машину/CI без правок кода:

    OIL_RENDER_SCALE   доля от нативного devicePixelRatio (1.0 = нативно)
    OIL_ANIM_FPS       частота кадров анимации/симуляции
    OIL_AA             режим сглаживания: fxaa | ssaa | msaa | none
    OIL_MSAA           число сэмплов MSAA (если OIL_AA=msaa)
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class RenderProfile:
    """Параметры рендера, подобранные под текущую ОС/железо.

    Поля заполняются в :meth:`detect`; реальное применение к плоттеру —
    в :meth:`apply` (там же поздняя деградация, если обнаружен software-GL).
    """

    os_name: str
    anti_aliasing: str | None  # 'fxaa' | 'ssaa' | 'msaa' | None
    multi_samples: int         # сэмплы для MSAA; 0 = выкл
    render_scale: float        # доля от нативного DPR (1.0 = нативно)
    animation_fps: int         # частота кадров анимации/симуляции
    desired_update_rate: float # целевой FPS LOD во время интеракции (VTK)
    still_update_rate: float   # целевой FPS в покое (VTK)
    particle_count: int        # частиц на систему (рычаг качества/нагрузки)

    # --------------------------------------------------------------- detect
    @classmethod
    def detect(cls) -> "RenderProfile":
        system = platform.system()

        # Разумные кросс-платформенные дефолты: FXAA дешевле MSAA×8 и ровнее
        # на HiDPI; render-on-demand делает высокий anim_fps безопасным.
        aa: str | None = "fxaa"
        msaa = 0
        scale = 1.0
        anim_fps = 60
        desired = 30.0
        still = 0.01
        particle_count = 760

        if system == "Darwin":
            # Retina рисует в 2-3× пикселей. Геометрия тут крошечная, поэтому
            # держим нативный масштаб, но оставляем рычаг через OIL_RENDER_SCALE.
            scale = 1.0
        elif system == "Windows":
            # Per-monitor DPI обычно ок; ничего особенного не требуется.
            pass
        elif system == "Linux":
            # Часто железо есть, но возможен software-GL (llvmpipe) по SSH/VNC.
            # Точную деградацию делаем в apply() по строке GL_RENDERER, здесь —
            # только дефолты.
            pass

        return cls(
            os_name=system,
            anti_aliasing=_env_str_aa(os.environ.get("OIL_AA"), aa),
            multi_samples=_env_int("OIL_MSAA", msaa),
            render_scale=_env_float("OIL_RENDER_SCALE", scale),
            animation_fps=_env_int("OIL_ANIM_FPS", anim_fps),
            desired_update_rate=desired,
            still_update_rate=still,
            particle_count=_env_int("OIL_PARTICLES", particle_count),
        )

    # ---------------------------------------------------------------- apply
    def apply(self, plotter) -> "RenderProfile":
        """Применить профиль к плоттеру; вернуть фактически применённый профиль.

        Может вернуть *деградированный* профиль, если после первого рендера
        обнаружен программный GL-рендер (нет GPU) — тогда отключаем сглаживание
        и снижаем частоту/масштаб, чтобы не упасть в 2-3 FPS на сервере.
        """
        effective = self
        if _is_software_gl(plotter):
            effective = self._degraded()

        effective._apply_anti_aliasing(plotter)
        effective._apply_update_rates(plotter)
        return effective

    def _apply_anti_aliasing(self, plotter) -> None:
        try:
            plotter.disable_anti_aliasing()
        except Exception:
            pass
        mode = self.anti_aliasing
        if not mode or mode == "none":
            return
        try:
            if mode == "msaa":
                plotter.enable_anti_aliasing("msaa", multi_samples=max(2, self.multi_samples or 4))
            else:
                plotter.enable_anti_aliasing(mode)
        except Exception:
            # FXAA/SSAA могут быть недоступны на отдельных драйверах — не падаем.
            pass

    def _apply_update_rates(self, plotter) -> None:
        rw = getattr(plotter, "render_window", None)
        if rw is None:
            return
        try:
            rw.SetDesiredUpdateRate(self.desired_update_rate)
        except Exception:
            pass

    def _degraded(self) -> "RenderProfile":
        from dataclasses import replace

        return replace(
            self,
            anti_aliasing=None,
            multi_samples=0,
            render_scale=min(self.render_scale, 0.85),
            animation_fps=min(self.animation_fps, 30),
            desired_update_rate=min(self.desired_update_rate, 12.0),
            particle_count=min(self.particle_count, 300),
        )


def _env_str_aa(value: str | None, default: str | None) -> str | None:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"fxaa", "ssaa", "msaa"}:
        return value
    if value in {"none", "off", "0", ""}:
        return None
    return default


def _is_software_gl(plotter) -> bool:
    """Эвристика: software-рендер (llvmpipe/swrast) — нет аппаратного GPU.

    Безопасна: при любой ошибке считаем, что GPU есть (не деградируем зря).
    """
    try:
        rw = plotter.render_window
        caps = rw.ReportCapabilities() or ""
    except Exception:
        return False
    caps = caps.lower()
    return any(token in caps for token in ("llvmpipe", "swrast", "software rasterizer"))
