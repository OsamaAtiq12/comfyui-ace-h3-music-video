"""Estimate ACE song duration from lyrics length (feeds encode + empty latent)."""

from __future__ import annotations

import math
import re


_SECTION_RE = re.compile(
    r"^\s*\[?\s*(intro|verse|pre[- ]?chorus|chorus|bridge|outro|hook|break|inst(?:rumental)?|solo|refrain)\s*\d*\s*\]?\s*$",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def estimate_duration_from_lyrics(
    lyrics: str,
    *,
    bpm: int = 120,
    sec_per_line: float = 2.7,
    words_per_sec: float = 2.0,
    pad_sec: float = 1.5,
    min_seconds: float = 8.0,
    max_seconds: float = 180.0,
    snap_to_bars: bool = True,
) -> float:
    """Heuristic sung-duration estimate from lyric structure.

    Prefers a tight fit (ACE often leaves a quiet tail if duration is too long).
    Trailing silence should still be trimmed after decode.
    """
    text = (lyrics or "").replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [ln.strip() for ln in text.split("\n")]

    lyric_lines: list[str] = []
    section_count = 0
    for ln in raw_lines:
        if not ln:
            continue
        if _SECTION_RE.match(ln):
            section_count += 1
            continue
        cleaned = re.sub(r"\(x\d+\)", "", ln, flags=re.IGNORECASE).strip()
        if cleaned:
            lyric_lines.append(cleaned)

    words = []
    for ln in lyric_lines:
        words.extend(_WORD_RE.findall(ln))

    line_est = len(lyric_lines) * float(sec_per_line)
    word_est = (len(words) / float(words_per_sec)) if words else 0.0
    # Small section bonus — avoid over-padding quiet outros
    section_est = section_count * 0.5

    # Average line/word estimates instead of taking the max (was overshooting)
    content = [v for v in (line_est, word_est) if v > 0]
    raw = (sum(content) / len(content) if content else float(min_seconds)) + float(pad_sec) + section_est
    if raw <= 0:
        raw = float(min_seconds)

    if snap_to_bars and bpm > 0:
        bar_sec = (60.0 / float(bpm)) * 4.0  # one bar in 4/4
        if bar_sec > 0:
            # Round to nearest bar (prefer slight under vs long silent tail)
            bars = max(1, int(round(raw / bar_sec)))
            raw = bars * bar_sec

    return float(max(float(min_seconds), min(float(max_seconds), raw)))


class EstimateSongDurationFromLyrics:
    """
    Auto song length from lyrics → wire into ACE duration + EmptyAceStep seconds.
    Video length follows the trimmed audio via chunking.
    Set manual_override_sec > 0 to force a fixed length instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "bpm": ("INT", {"default": 120, "min": 40, "max": 240}),
                "sec_per_line": (
                    "FLOAT",
                    {
                        "default": 2.7,
                        "min": 1.0,
                        "max": 8.0,
                        "step": 0.1,
                        "tooltip": "Seconds of music assumed per lyric line.",
                    },
                ),
                "words_per_sec": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.5,
                        "max": 6.0,
                        "step": 0.1,
                        "tooltip": "Sung words per second (higher = shorter song).",
                    },
                ),
                "pad_sec": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "Small intro/outro pad. Keep low to avoid silent tails.",
                    },
                ),
                "min_seconds": ("FLOAT", {"default": 8.0, "min": 2.0, "max": 60.0, "step": 0.5}),
                "max_seconds": ("FLOAT", {"default": 180.0, "min": 8.0, "max": 600.0, "step": 1.0}),
                "snap_to_bars": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Snap to nearest 4/4 bars at BPM."},
                ),
                "manual_override_sec": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 600.0,
                        "step": 0.5,
                        "tooltip": "If > 0, use this fixed duration instead of lyrics estimate.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("duration_sec", "summary")
    FUNCTION = "estimate"
    CATEGORY = "audio/music_video"

    def estimate(
        self,
        lyrics: str,
        bpm: int,
        sec_per_line: float,
        words_per_sec: float,
        pad_sec: float,
        min_seconds: float,
        max_seconds: float,
        snap_to_bars: bool,
        manual_override_sec: float,
    ):
        if float(manual_override_sec) > 0:
            dur = float(manual_override_sec)
            summary = f"manual override -> {dur:.1f}s"
        else:
            dur = estimate_duration_from_lyrics(
                lyrics,
                bpm=int(bpm),
                sec_per_line=float(sec_per_line),
                words_per_sec=float(words_per_sec),
                pad_sec=float(pad_sec),
                min_seconds=float(min_seconds),
                max_seconds=float(max_seconds),
                snap_to_bars=bool(snap_to_bars),
            )
            lines = [
                ln.strip()
                for ln in (lyrics or "").splitlines()
                if ln.strip() and not _SECTION_RE.match(ln.strip())
            ]
            words = len(_WORD_RE.findall(lyrics or ""))
            summary = (
                f"auto from lyrics -> {dur:.1f}s "
                f"({len(lines)} lines, {words} words, {bpm} bpm)"
            )

        print(f"[ACE duration] {summary}")
        return {
            "ui": {"text": [summary]},
            "result": (float(dur), summary),
        }
