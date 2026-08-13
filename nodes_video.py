from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import folder_paths
import numpy as np
from PIL import Image

from .ffmpeg_util import find_ffmpeg


class ExtractLastFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "extract"
    CATEGORY = "video/music_video"

    def extract(self, images):
        return (images[-1:].clone(),)


class SaveVideoFramesNoAudio:
    """Encode IMAGE frames to MP4 with no audio track (discards H3 audio by design)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "music_video/clips/clip"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                "exact_filename": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "If set (e.g. clip_000.mp4), write exactly this name under the prefix folder.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save"
    CATEGORY = "video/music_video"
    OUTPUT_NODE = True

    def save(self, images, filename_prefix: str, fps: float, exact_filename: str = ""):
        ffmpeg = find_ffmpeg()
        output_dir = Path(folder_paths.get_output_directory())

        if exact_filename.strip():
            # Stable path for orchestrator: output/<prefix_dir>/<exact_filename>
            # Treat filename_prefix as a directory when it ends with / or looks like a folder.
            prefix = filename_prefix.strip().replace("\\", "/").rstrip("/")
            # If prefix looks like ".../clip", use its parent as the folder.
            prefix_path = Path(prefix)
            if prefix_path.name.startswith("clip") and prefix_path.suffix == "":
                folder = output_dir / prefix_path.parent
            else:
                folder = output_dir / prefix_path
            folder.mkdir(parents=True, exist_ok=True)
            out_path = folder / exact_filename.strip()
        else:
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, str(output_dir), images.shape[2], images.shape[1]
            )
            folder = Path(full_output_folder)
            out_path = folder / f"{filename}_{counter:05}_.mp4"

        with tempfile.TemporaryDirectory(prefix="comfy_mv_frames_") as td:
            td = Path(td)
            frames_dir = td / "frames"
            frames_dir.mkdir()
            imgs = images.detach().cpu().numpy()
            for i, frame in enumerate(imgs):
                arr = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(arr).save(frames_dir / f"f_{i:06d}.png")

            cmd = [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "f_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-an",
                str(out_path),
            ]
            subprocess.run(cmd, check=True, capture_output=True)

        return {
            "ui": {"text": [str(out_path)]},
            "result": (str(out_path),),
        }


class AssembleMusicVideoFFmpeg:
    """
    Concatenate silent H3 clips in order, mux original ACE-Step track.wav.
    H3 per-clip audio is never used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_path": ("STRING", {"default": ""}),
                "clips_dir": (
                    "STRING",
                    {
                        "default": "music_video/clips",
                        "tooltip": "Folder (under output/ or absolute) containing clip_000.mp4 …",
                    },
                ),
                "output_relative_path": ("STRING", {"default": "music_video/music_video.mp4"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
            },
            "optional": {
                "track_path_override": (
                    "STRING",
                    {"default": "", "tooltip": "Optional override; default uses manifest track_path."},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_path",)
    FUNCTION = "assemble"
    CATEGORY = "video/music_video"
    OUTPUT_NODE = True

    def assemble(
        self,
        manifest_path: str,
        clips_dir: str,
        output_relative_path: str,
        fps: float,
        track_path_override: str = "",
    ):
        ffmpeg = find_ffmpeg()
        out_root = Path(folder_paths.get_output_directory())

        mp = Path(manifest_path.strip())
        if not mp.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        data = json.loads(mp.read_text(encoding="utf-8"))
        chunks = data["chunks"]
        track = Path(track_path_override.strip() or data["track_path"])
        if not track.is_file():
            raise FileNotFoundError(f"ACE-Step track not found: {track}")

        cdir = Path(clips_dir.strip())
        if not cdir.is_dir():
            alt = out_root / clips_dir.strip().lstrip("/\\")
            if alt.is_dir():
                cdir = alt
            else:
                raise FileNotFoundError(f"Clips dir not found: {clips_dir}")

        final_path = out_root / output_relative_path.strip().lstrip("/\\")
        if not output_relative_path.strip() or final_path.resolve() == out_root.resolve() or final_path.is_dir():
            final_path = out_root / "music_video" / "music_video.mp4"
        final_path.parent.mkdir(parents=True, exist_ok=True)

        def _run(cmd: list[str], label: str) -> None:
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                err = (r.stderr or r.stdout or "").strip()
                raise RuntimeError(
                    f"ffmpeg {label} failed (exit {r.returncode}):\n{err[-4000:]}\nCMD: {' '.join(cmd)}"
                )

        with tempfile.TemporaryDirectory(prefix="comfy_mv_assemble_") as td:
            td = Path(td)
            timed = []
            for entry in chunks:
                src = cdir / entry["clip_name"]
                if not src.is_file():
                    raise FileNotFoundError(f"Missing clip: {src}")
                duration = float(entry["duration_sec"])
                out_clip = td / entry["clip_name"]
                # Force exact chunk duration (trim or pad last frame); strip any audio
                cmd = [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(src),
                    "-an",
                    "-vf",
                    f"fps={fps}",
                    "-t",
                    f"{duration:.6f}",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "18",
                    str(out_clip),
                ]
                ffprobe = Path(ffmpeg).with_name(
                    "ffprobe.exe" if Path(ffmpeg).name.lower() == "ffmpeg.exe" else "ffprobe"
                )
                probe = subprocess.run(
                    [
                        str(ffprobe),
                        "-v",
                        "error",
                        "-show_entries",
                        "format=duration",
                        "-of",
                        "default=noprint_wrappers=1:nokey=1",
                        str(src),
                    ],
                    capture_output=True,
                    text=True,
                )
                clip_dur = float(probe.stdout.strip() or 0.0) if probe.returncode == 0 else duration
                if clip_dur + 0.05 < duration:
                    pad = duration - clip_dur
                    cmd = [
                        ffmpeg,
                        "-y",
                        "-i",
                        str(src),
                        "-an",
                        "-vf",
                        f"fps={fps},tpad=stop_mode=clone:stop_duration={pad:.6f}",
                        "-t",
                        f"{duration:.6f}",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-crf",
                        "18",
                        str(out_clip),
                    ]
                _run(cmd, f"normalize {entry['clip_name']}")
                timed.append(out_clip)

            list_file = td / "concat.txt"
            # Windows-safe concat list: forward slashes, escaped single quotes
            lines = []
            for p in timed:
                path = p.resolve().as_posix().replace("'", r"'\''")
                lines.append(f"file '{path}'\n")
            list_file.write_text("".join(lines), encoding="utf-8")
            silent_concat = td / "silent.mp4"
            _run(
                [
                    ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_file),
                    "-c",
                    "copy",
                    str(silent_concat),
                ],
                "concat clips",
            )

            total_dur = float(data["total_duration_sec"])
            # Mux ACE track; trim/pad audio to video length
            _run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(silent_concat),
                    "-i",
                    str(track),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-shortest",
                    "-t",
                    f"{total_dur:.6f}",
                    str(final_path),
                ],
                "mux ACE audio",
            )

        try:
            rel_parent = final_path.parent.resolve().relative_to(out_root.resolve())
            subfolder = str(rel_parent).replace("\\", "/")
            if subfolder == ".":
                subfolder = ""
        except ValueError:
            subfolder = "music_video"

        media = {
            "filename": final_path.name,
            "subfolder": subfolder,
            "type": "output",
            "format": "video/mp4",
            "frame_rate": 24,
        }
        return {
            "ui": {
                "text": [f"Final music video: {final_path}"],
                "images": [media],
                "animated": [True],
                "gifs": [media],
            },
            "result": (str(final_path),),
        }


class PreviewFinalMusicVideo:
    """
    Final output node: preview the assembled music video in the ComfyUI UI
    and expose VIDEO + path for Save Video / further nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Absolute or output-relative path from Assemble Music Video.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "preview"
    CATEGORY = "video/music_video"
    OUTPUT_NODE = True

    def preview(self, video_path: str):
        out_root = Path(folder_paths.get_output_directory()).resolve()
        raw = (video_path or "").strip()
        if not raw:
            raise FileNotFoundError("PreviewFinalMusicVideo: empty video_path")

        p = Path(raw)
        if not p.is_file():
            alt = out_root / raw.lstrip("/\\")
            if alt.is_file():
                p = alt
            else:
                raise FileNotFoundError(f"Final music video not found: {video_path}")

        p = p.resolve()
        try:
            rel = p.relative_to(out_root)
            subfolder = str(rel.parent).replace("\\", "/")
            if subfolder == ".":
                subfolder = ""
            filename = rel.name
            ui_type = "output"
        except ValueError:
            filename = p.name
            subfolder = "music_video"
            ui_type = "output"
            dest = out_root / subfolder / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.resolve() != p:
                shutil.copy2(p, dest)
            p = dest

        media = {
            "filename": filename,
            "subfolder": subfolder,
            "type": ui_type,
            "format": "video/mp4",
            "frame_rate": 24,
        }
        return {
            "ui": {
                "text": [f"Final output: {p}"],
                # Core SaveVideo style
                "images": [media],
                "animated": [True],
                # VHS / custom-node style (picked up by web/preview_final.js)
                "gifs": [media],
            },
            "result": (str(p),),
        }
