from __future__ import annotations

import json
import wave
from pathlib import Path

import folder_paths
import numpy as np
import torch

from .ffmpeg_util import h3_frame_length


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _audio_to_wav_pcm(audio: dict) -> tuple[np.ndarray, int]:
    """Return int16 interleaved PCM [T, C] and sample rate."""
    waveform = audio["waveform"]  # [B, C, T]
    sr = int(audio["sample_rate"])
    wav = waveform[0].detach().cpu().float().numpy()
    if wav.ndim == 1:
        wav = wav[None, :]
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav.T * 32767.0).astype(np.int16)  # [T, C]
    return np.ascontiguousarray(pcm), sr


def _write_wav(path: Path, pcm: np.ndarray, sr: int) -> None:
    _ensure_parent(path)
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _load_audio_file(path: Path) -> dict:
    """Load WAV via stdlib wave — avoids torchaudio's TorchCodec dependency."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_width == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} ({path})")

    if channels > 1:
        pcm = pcm.reshape(-1, channels).T  # [C, T]
    else:
        pcm = pcm.reshape(1, -1)

    waveform = torch.from_numpy(np.ascontiguousarray(pcm))
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(sample_rate)}


def _trim_silence_audio(
    audio: dict,
    *,
    threshold: float = 0.02,
    min_silence_sec: float = 0.35,
    pad_end_sec: float = 0.15,
    trim_leading: bool = True,
) -> dict:
    """Trim trailing (and optional leading) near-silence from an AUDIO dict."""
    waveform = audio["waveform"]  # [B, C, T]
    sr = int(audio["sample_rate"])
    wav = waveform[0].detach().cpu().float()
    if wav.ndim == 1:
        mono = wav.abs()
    else:
        mono = wav.abs().mean(dim=0)

    if mono.numel() == 0:
        return audio

    # Smooth envelope so short breaths don't stop the trim
    win = max(1, int(sr * 0.02))
    kernel = torch.ones(win, dtype=torch.float32) / float(win)
    padded = torch.nn.functional.pad(mono.view(1, 1, -1), (win // 2, win - 1 - win // 2), mode="reflect")
    env = torch.nn.functional.conv1d(padded, kernel.view(1, 1, -1)).view(-1)
    active = env > float(threshold)

    if not bool(active.any()):
        return audio

    idx = torch.where(active)[0]
    start = int(idx[0].item()) if trim_leading else 0
    end = int(idx[-1].item()) + 1

    # Only trim trailing silence if it's long enough to matter
    trailing = mono.numel() - end
    if trailing < int(float(min_silence_sec) * sr):
        end = mono.numel()
    else:
        end = min(mono.numel(), end + int(float(pad_end_sec) * sr))

    if trim_leading:
        leading = start
        if leading < int(0.2 * sr):
            start = 0
        else:
            start = max(0, start - int(0.05 * sr))

    if start <= 0 and end >= mono.numel():
        return audio

    trimmed = waveform[:, :, start:end].contiguous()
    return {"waveform": trimmed, "sample_rate": sr}


class TrimTrailingSilence:
    """Remove quiet tails (and optional heads) so video length matches real music."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "threshold": (
                    "FLOAT",
                    {
                        "default": 0.02,
                        "min": 0.001,
                        "max": 0.2,
                        "step": 0.001,
                        "tooltip": "RMS-ish envelope threshold; higher = more aggressive trim.",
                    },
                ),
                "min_silence_sec": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.05,
                        "max": 5.0,
                        "step": 0.05,
                        "tooltip": "Only trim a trailing quiet region at least this long.",
                    },
                ),
                "pad_end_sec": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "trim_leading": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio", "duration_sec", "trimmed_sec")
    FUNCTION = "trim"
    CATEGORY = "audio/music_video"

    def trim(self, audio, threshold, min_silence_sec, pad_end_sec, trim_leading):
        before = float(audio["waveform"].shape[-1]) / float(audio["sample_rate"])
        out = _trim_silence_audio(
            audio,
            threshold=float(threshold),
            min_silence_sec=float(min_silence_sec),
            pad_end_sec=float(pad_end_sec),
            trim_leading=bool(trim_leading),
        )
        after = float(out["waveform"].shape[-1]) / float(out["sample_rate"])
        trimmed = max(0.0, before - after)
        print(f"[TrimSilence] {before:.2f}s -> {after:.2f}s (removed {trimmed:.2f}s)")
        return (out, after, trimmed)


class SaveAudioTrackWav:
    """Persist ACE-Step (or any) AUDIO as a WAV under the output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "relative_path": ("STRING", {"default": "audio/track.wav"}),
            },
        }

    RETURN_TYPES = ("STRING", "AUDIO", "FLOAT")
    RETURN_NAMES = ("track_path", "audio", "duration_sec")
    FUNCTION = "save"
    CATEGORY = "audio/music_video"
    OUTPUT_NODE = True

    def save(self, audio, relative_path: str):
        out_root = Path(folder_paths.get_output_directory())
        rel = relative_path.strip().lstrip("/\\")
        path = out_root / rel
        pcm, sr = _audio_to_wav_pcm(audio)
        _write_wav(path, pcm, sr)
        duration = float(pcm.shape[0]) / float(sr)
        return {
            "ui": {"text": [f"Saved {path} ({duration:.2f}s)"]},
            "result": (str(path), audio, duration),
        }


class ChunkAudioForMiniMaxH3:
    """Split a full track into consecutive ≤15s windows for H3 reference audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "chunk_seconds": (
                    "FLOAT",
                    {
                        "default": 12.0,
                        "min": 2.0,
                        "max": 15.0,
                        "step": 0.1,
                        "tooltip": "H3 reference audio must be 2–15s; default 12s leaves headroom.",
                    },
                ),
                "output_subdir": ("STRING", {"default": "audio/chunks"}),
                "manifest_name": ("STRING", {"default": "manifest.json"}),
                "track_relative_path": ("STRING", {"default": "audio/track.wav"}),
                "also_save_full_track": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("manifest_path", "track_path", "chunk_count", "total_duration_sec")
    FUNCTION = "chunk"
    CATEGORY = "audio/music_video"

    def chunk(
        self,
        audio,
        chunk_seconds: float,
        output_subdir: str,
        manifest_name: str,
        track_relative_path: str,
        also_save_full_track: bool,
    ):
        chunk_seconds = float(chunk_seconds)
        if chunk_seconds > 15.0:
            raise ValueError("chunk_seconds cannot exceed 15 (MiniMax H3 reference-audio cap).")
        if chunk_seconds < 2.0:
            raise ValueError("chunk_seconds must be at least 2 seconds.")

        out_root = Path(folder_paths.get_output_directory())
        chunk_dir = out_root / output_subdir.strip().lstrip("/\\")
        chunk_dir.mkdir(parents=True, exist_ok=True)

        pcm, sr = _audio_to_wav_pcm(audio)
        total_samples = int(pcm.shape[0])
        total_duration = total_samples / float(sr)
        chunk_samples = max(1, int(round(chunk_seconds * sr)))

        track_path = out_root / track_relative_path.strip().lstrip("/\\")
        if also_save_full_track:
            _write_wav(track_path, pcm, sr)

        chunks = []
        start = 0
        index = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            # Avoid a tiny leftover tail: merge into previous if < 2s and we already have chunks
            remaining = total_samples - start
            if index > 0 and remaining < int(2.0 * sr) and remaining < chunk_samples:
                # extend previous chunk instead
                prev = chunks[-1]
                prev_pcm = pcm[int(prev["start_sample"]) : total_samples]
                prev_path = Path(prev["path"])
                _write_wav(prev_path, prev_pcm, sr)
                prev["end_sec"] = total_duration
                prev["end_sample"] = total_samples
                prev["duration_sec"] = prev["end_sec"] - prev["start_sec"]
                prev["frame_length"] = h3_frame_length(prev["duration_sec"])
                break

            piece = pcm[start:end]
            rel_name = f"chunk_{index:03d}.wav"
            path = chunk_dir / rel_name
            _write_wav(path, piece, sr)
            start_sec = start / float(sr)
            end_sec = end / float(sr)
            duration = end_sec - start_sec
            entry = {
                "index": index,
                "path": str(path),
                "relative_path": str(Path(output_subdir) / rel_name).replace("\\", "/"),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": duration,
                "start_sample": start,
                "end_sample": end,
                "sample_rate": sr,
                "frame_length": h3_frame_length(duration),
                "clip_name": f"clip_{index:03d}.mp4",
            }
            chunks.append(entry)
            index += 1
            start = end

        manifest = {
            "version": 1,
            "track_path": str(track_path),
            "chunk_seconds_requested": chunk_seconds,
            "sample_rate": sr,
            "total_duration_sec": total_duration,
            "chunk_count": len(chunks),
            "fps": 24,
            "notes": (
                "MiniMax H3: ref audio 2–15s, total ref audio ≤15s per call, "
                "requires ≥1 ref image or video with audio; output video ≤15s. "
                "Final mux must use track_path (ACE-Step), not H3 per-clip audio."
            ),
            "chunks": chunks,
        }
        manifest_path = chunk_dir / manifest_name
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        return {
            "ui": {"text": [f"{len(chunks)} chunks -> {manifest_path}"]},
            "result": (str(manifest_path), str(track_path), len(chunks), total_duration),
        }


class LoadAudioFromPath:
    """Load a WAV/FLAC/MP3 from an absolute or output-relative path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("audio", "duration_sec")
    FUNCTION = "load"
    CATEGORY = "audio/music_video"

    def load(self, path: str):
        p = Path(path.strip())
        if not p.is_file():
            # try under output directory
            alt = Path(folder_paths.get_output_directory()) / path.strip().lstrip("/\\")
            if alt.is_file():
                p = alt
            else:
                raise FileNotFoundError(f"Audio not found: {path}")
        audio = _load_audio_file(p)
        n = int(audio["waveform"].shape[-1])
        duration = n / float(audio["sample_rate"])
        return (audio, duration)


class LoadManifestAudioChunk:
    """Load chunk N from a ChunkAudioForMiniMaxH3 manifest."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_path": ("STRING", {"default": ""}),
                "chunk_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            },
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "audio",
        "start_sec",
        "end_sec",
        "duration_sec",
        "frame_length",
        "chunk_path",
        "clip_name",
    )
    FUNCTION = "load"
    CATEGORY = "audio/music_video"

    def load(self, manifest_path: str, chunk_index: int):
        mp = Path(manifest_path.strip())
        if not mp.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        data = json.loads(mp.read_text(encoding="utf-8"))
        chunks = data["chunks"]
        if chunk_index < 0 or chunk_index >= len(chunks):
            raise IndexError(f"chunk_index {chunk_index} out of range (0..{len(chunks)-1})")
        entry = chunks[chunk_index]
        audio = _load_audio_file(Path(entry["path"]))
        return (
            audio,
            float(entry["start_sec"]),
            float(entry["end_sec"]),
            float(entry["duration_sec"]),
            int(entry["frame_length"]),
            str(entry["path"]),
            str(entry["clip_name"]),
        )


class SecondsToH3FrameLength:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seconds": ("FLOAT", {"default": 12.0, "min": 0.2, "max": 15.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "convert"
    CATEGORY = "audio/music_video"

    def convert(self, seconds: float, fps: float):
        return (h3_frame_length(seconds, fps=fps),)
