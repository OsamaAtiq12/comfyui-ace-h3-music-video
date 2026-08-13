"""Generate music via bodycamhooker's local ACE-Step 1.5 API (:8001).

These weights are HuggingFace ACE-Step packs (not Comfy-Org converted files):
  E:\\judge reacts\\bodycamhooker\\models\\Ace-Step\\checkpoints
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import folder_paths

from .ffmpeg_util import find_ffmpeg
from .nodes_audio import _load_audio_file

DEFAULT_ACE_API = "http://127.0.0.1:8001"
DEFAULT_CHECKPOINTS = r"E:\judge reacts\bodycamhooker\models\Ace-Step\checkpoints"


def _request_json(method: str, url: str, payload: dict | None = None, timeout: float = 180.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ensure_wav(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".wav":
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest
    ffmpeg = find_ffmpeg()
    subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-acodec", "pcm_s16le", "-ar", "48000", str(dest)],
        check=True,
        capture_output=True,
    )
    return dest


def generate_ace_track(
    *,
    tags: str,
    lyrics: str = "",
    duration_sec: float = 60.0,
    bpm: int = 120,
    key_scale: str = "E minor",
    vocal_language: str = "en",
    thinking: bool = True,
    api_url: str = DEFAULT_ACE_API,
    relative_out: str = "audio/track.wav",
    timeout_sec: float = 900.0,
) -> tuple[str, dict, float]:
    """Call ACE-Step API, save WAV under Comfy output/, return (path, AUDIO dict, duration)."""
    base = api_url.rstrip("/")
    try:
        health = _request_json("GET", f"{base}/health", timeout=5.0)
    except Exception as exc:
        raise RuntimeError(
            f"ACE-Step API not reachable at {base}. "
            r"Start it with: E:\judge reacts\bodycamhooker\start-dev.ps1 "
            f"(weights under {DEFAULT_CHECKPOINTS}). ({exc})"
        ) from exc

    status = ((health.get("data") or {}).get("status") or "")
    if status and status != "ok":
        raise RuntimeError(f"ACE-Step API unhealthy: {status}")

    payload = {
        "task_type": "text2music",
        "thinking": bool(thinking),
        "sample_mode": False,
        "instrumental": not bool(lyrics.strip()),
        "vocal_language": vocal_language or "en",
        "batch_size": 1,
        "audio_format": "wav",
        "prompt": tags.strip() or "cinematic indie rock",
        "audio_duration": float(duration_sec),
        "bpm": int(bpm),
        "use_format": bool(lyrics.strip()),
    }
    if lyrics.strip():
        payload["lyrics"] = lyrics
    if key_scale.strip():
        payload["key_scale"] = key_scale.strip()

    body = _request_json("POST", f"{base}/release_task", payload, timeout=180.0)
    task_id = (body.get("data") or {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"ACE-Step did not return task_id: {body}")

    deadline = time.time() + timeout_sec
    file_ref = None
    while time.time() < deadline:
        q = _request_json(
            "POST",
            f"{base}/query_result",
            {"task_id_list": [str(task_id)]},
            timeout=120.0,
        )
        items = q.get("data") or []
        if not items:
            time.sleep(2.0)
            continue
        item = items[0]
        st = item.get("status")
        if st in (2, "2", "failed"):
            raise RuntimeError(item.get("error") or item.get("message") or "ACE-Step failed")
        if st not in (1, "1", "succeeded"):
            time.sleep(2.0)
            continue

        raw = item.get("result")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = None
        entries = raw if isinstance(raw, list) else ([raw] if isinstance(raw, dict) else [])
        if not entries:
            time.sleep(2.0)
            continue
        first = entries[0]
        for key in ("file", "path", "first_audio_path"):
            if first.get(key):
                file_ref = str(first[key])
                break
        if not file_ref:
            paths = first.get("audio_paths") or first.get("raw_audio_paths") or []
            if paths:
                file_ref = str(paths[0])
        if file_ref:
            break
        time.sleep(2.0)

    if not file_ref:
        raise RuntimeError(f"Timed out waiting for ACE-Step task {task_id}")

    out_root = Path(folder_paths.get_output_directory())
    dest = out_root / relative_out.strip().lstrip("/\\")
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Download to temp then ensure wav
    tmp = dest.with_suffix(".download" + (Path(file_ref).suffix or ".bin"))
    if file_ref.startswith("http://") or file_ref.startswith("https://"):
        url = file_ref
    elif file_ref.startswith("/"):
        url = f"{base}{file_ref}"
    else:
        url = f"{base}/v1/audio?{urllib.parse.urlencode({'path': file_ref})}"

    # Prefer local path if API returned an absolute filesystem path
    local = Path(file_ref)
    if local.is_file():
        _ensure_wav(local, dest)
    else:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            tmp.write_bytes(resp.read())
        _ensure_wav(tmp, dest)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    audio = _load_audio_file(dest)
    duration = float(audio["waveform"].shape[-1]) / float(audio["sample_rate"])
    return str(dest), audio, duration


class AceStep15GenerateViaAPI:
    """Text/lyrics → full song via bodycamhooker ACE-Step 1.5 API (local open weights)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {"multiline": True, "default": "cinematic indie rock, anthemic chorus"}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "duration_sec": ("FLOAT", {"default": 60.0, "min": 5.0, "max": 360.0, "step": 1.0}),
                "bpm": ("INT", {"default": 120, "min": 30, "max": 300}),
                "key_scale": ("STRING", {"default": "E minor"}),
                "vocal_language": ("STRING", {"default": "en"}),
                "thinking": ("BOOLEAN", {"default": True}),
                "api_url": ("STRING", {"default": DEFAULT_ACE_API}),
                "relative_out": ("STRING", {"default": "audio/track.wav"}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING", "FLOAT")
    RETURN_NAMES = ("audio", "track_path", "duration_sec")
    FUNCTION = "generate"
    CATEGORY = "audio/music_video"
    OUTPUT_NODE = True

    def generate(
        self,
        tags,
        lyrics,
        duration_sec,
        bpm,
        key_scale,
        vocal_language,
        thinking,
        api_url,
        relative_out,
    ):
        path, audio, duration = generate_ace_track(
            tags=tags,
            lyrics=lyrics,
            duration_sec=duration_sec,
            bpm=bpm,
            key_scale=key_scale,
            vocal_language=vocal_language,
            thinking=thinking,
            api_url=api_url,
            relative_out=relative_out,
        )
        return {
            "ui": {"text": [f"ACE-Step -> {path} ({duration:.1f}s)"]},
            "result": (audio, path, duration),
        }
