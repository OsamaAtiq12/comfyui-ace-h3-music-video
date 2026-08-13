from __future__ import annotations

import shutil
from pathlib import Path

FFMPEG_CANDIDATES = [
    Path(r"D:\WanGp\ffmpeg_bins\ffmpeg.exe"),
    Path(shutil.which("ffmpeg") or ""),
]


def find_ffmpeg() -> str:
    for p in FFMPEG_CANDIDATES:
        if p and p.is_file():
            return str(p)
    raise FileNotFoundError(
        "ffmpeg not found. Expected at D:\\WanGp\\ffmpeg_bins\\ffmpeg.exe or on PATH."
    )


def h3_frame_length(seconds: float, fps: float = 24.0, max_frames: int = 362) -> int:
    """Snap duration to MiniMax H3's 17k+5 frame grid (24 fps). Cap at ~15.08s (362)."""
    seconds = max(0.2, float(seconds))
    target = int(round(seconds * fps))
    # smallest n >= target with n = 17k + 5
    n = max(5, target)
    rem = (n - 5) % 17
    if rem != 0:
        n += 17 - rem
    return min(n, max_frames)
