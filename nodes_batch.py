"""Batch MiniMax H3 Ref2VA over audio chunks from a manifest, then ready for assemble."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import folder_paths
import numpy as np
import torch
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
from PIL import Image
from comfy_extras.nodes_custom_sampler import Guider_Basic, Noise_RandomNoise, BasicScheduler
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo, MiniMaxH3SigmaShift
from nodes import VAEDecode

from .ffmpeg_util import find_ffmpeg, h3_frame_length
from .nodes_audio import _load_audio_file
from .nodes_video import SaveVideoFramesNoAudio, AssembleMusicVideoFFmpeg

# Shot 0: character sheet = person only (<Picture 1>) + audio (<Audio 1>)
DEFAULT_PROMPT = """subject_definitions:
<Subject 1> is the young woman from <Picture 1> only — face, hair, skin, body proportions, and wardrobe. Ignore the background, set, lighting, and camera framing of <Picture 1>.
<Audio 1> is the song she is performing.

summary:
[reference generation] Premium live-action music video of <Subject 1> singing to <Audio 1> in a brand-new cinematic location. Use <Picture 1> only as character identity, never as the scene.

retention_analysis:
Fully preserve <Subject 1> identity and outfit from <Picture 1>. Do not copy the character-sheet background, studio backdrop, flat wall, seaside pavilion, rooftop, or prior set. Invent the rainy neon street environment below.

detailed_description: [Shot 1] Cinematic live-action music video, photoreal. Medium close-up of <Subject 1> standing center-frame, singing into the camera with clear lip sync to <Audio 1>, natural micro-expressions, light rain mist on hair and shoulders, gentle sway to the beat. Brand-new background: rainy night city street lined with colorful neon signs — magenta, cyan, and warm amber reflections on wet asphalt, soft umbrella lights and shop windows bokeh behind her, light rain droplets in air, shallow depth of field, moody cinematic atmosphere. Clean beauty key light on her face so she stays readable against the neon. Camera: slow Push In with small amplitude at slow speed. Continuous single shot, no hard cut, no identity drift, no outfit change, no plain empty background, no character-sheet backdrop, no golden-hour seaside glass pavilion.

overall_soundscape: Follow reference <Audio 1> closely (diegetic performance energy). Soft rain and distant city night ambience under the vocal.
non_diegetic_music: Match <Audio 1> timing and dynamics; do not invent a different song.
"""

# Later shots: <Picture 1> = person only, <Picture 2> = last frame of previous clip
DEFAULT_PROMPT_CONTINUITY = """subject_definitions:
<Subject 1> is the young woman from <Picture 1> only — face, hair, skin, body proportions, and wardrobe. Ignore the background of <Picture 1>.
<Picture 2> is the ending frame of the previous clip and is the mandatory starting pose, framing, lighting, and background for this shot.
<Audio 1> is the song she is performing.

summary:
[reference generation] Continue the same rainy neon street music video seamlessly from <Picture 2>, keeping <Subject 1> locked to the character sheet.

retention_analysis:
Fully preserve <Subject 1> identity/outfit from <Picture 1>. Fully preserve pose, framing, lighting, and the rainy neon street set from <Picture 2> at the start. Do not revert to the character-sheet background or switch to a seaside pavilion / rooftop set.

detailed_description: [Shot 1] Cinematic live-action music video, photoreal. Begin exactly matching <Picture 2> (pose, camera, rainy neon street, wet reflections). Keep <Subject 1> identity from <Picture 1>. She continues singing to <Audio 1> with natural lip sync, soft motion, and light rain mist; neon bokeh and wet asphalt reflections stay rich behind her. Camera: soft Push In with small amplitude at slow speed. Continuous single shot, no jump cut, no identity drift, no outfit change.

overall_soundscape: Follow reference <Audio 1> closely. Soft rain and distant city night ambience under the vocal.
non_diegetic_music: Match <Audio 1> timing and dynamics; do not invent a different song.
"""


def _load_audio(path: Path) -> dict:
    return _load_audio_file(path)


def _unpack(node_output):
    """Unpack Comfy NodeOutput / tuple into a list of values."""
    if hasattr(node_output, "__getitem__"):
        out = []
        i = 0
        while True:
            try:
                out.append(node_output[i])
                i += 1
            except Exception:
                break
        if out:
            return out
    if isinstance(node_output, (tuple, list)):
        return list(node_output)
    return [node_output]


def _pil_to_image_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]  # [1, H, W, 3]


def _load_last_frame_from_video(video_path: Path) -> torch.Tensor | None:
    """Extract the last decoded frame of an MP4 as an IMAGE tensor [1,H,W,3]."""
    if not video_path.is_file():
        return None
    ffmpeg = find_ffmpeg()
    with tempfile.TemporaryDirectory(prefix="comfy_mv_last_") as td:
        out_png = Path(td) / "last.png"
        # -sseof seeks near end; -update 1 writes a single still
        cmd = [
            ffmpeg,
            "-y",
            "-sseof",
            "-0.15",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(out_png),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            # Fallback: read from start and take last via a longer decode if needed
            cmd2 = [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-vf",
                "select=eq(n\\,0)",
                "-vsync",
                "vfr",
                "-frames:v",
                "1",
                str(out_png),
            ]
            # Last-frame via reverse is expensive; try sseof again with larger window
            cmd2 = [
                ffmpeg,
                "-y",
                "-sseof",
                "-1",
                "-i",
                str(video_path),
                "-update",
                "1",
                "-q:v",
                "2",
                str(out_png),
            ]
            try:
                subprocess.run(cmd2, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                print(f"[H3 batch] warning: could not extract last frame from {video_path}: {e}")
                return None
        if not out_png.is_file():
            return None
        return _pil_to_image_tensor(Image.open(out_png))


def _save_continuity_still(frame: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (frame[0].detach().cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def _resolve_prompts(prompt: str, continuity_prompt: str) -> tuple[str, str]:
    first = (prompt or "").strip() or DEFAULT_PROMPT.strip()
    cont = (continuity_prompt or "").strip()
    if not cont:
        # If user only edited the first prompt and omitted Picture 2 language,
        # fall back to the strong continuity template.
        if "<Picture 2>" in first:
            cont = first
        else:
            cont = DEFAULT_PROMPT_CONTINUITY.strip()
    return first, cont


class MiniMaxH3MusicVideoBatch:
    """
    Full Stage 4: for each manifest chunk, run MiniMax H3 Ref2VA, save silent clip_XXX.mp4.
    With use_continuity=True (default), each next clip gets the previous clip's last frame
    as <Picture 2> so pose/framing/lighting stay continuous across the music video.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE", {"tooltip": "H3 video VAE"}),
                "audio_vae": ("VAE", {"tooltip": "H3 audio VAE (Comfy-Org)"}),
                "ref_image": ("IMAGE", {"tooltip": "Style/character reference (required with audio)"}),
                "manifest_path": ("STRING", {"default": ""}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_PROMPT,
                        "tooltip": "Prompt for the FIRST chunk (<Picture 1> + <Audio 1>).",
                    },
                ),
                "continuity_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": DEFAULT_PROMPT_CONTINUITY,
                        "tooltip": "Prompt for later chunks. Must reference <Picture 2> as the previous ending frame.",
                    },
                ),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 432, "min": 64, "max": 4096, "step": 32}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 50, "tooltip": "15 for quality; 4 is fastest with Turbo LoRA."}),
                "rng": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sampler_name": (comfy.samplers.SAMPLER_NAMES, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "simple"}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "use_continuity": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Pass last frame of clip N as <Picture 2> when generating clip N+1.",
                    },
                ),
                "clips_subdir": ("STRING", {"default": "music_video/clips"}),
                "skip_existing": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("clips_dir", "manifest_path", "clip_count")
    FUNCTION = "run"
    CATEGORY = "video/music_video"

    def run(
        self,
        model,
        clip,
        vae,
        audio_vae,
        ref_image,
        manifest_path,
        prompt,
        continuity_prompt,
        width,
        height,
        steps,
        rng,
        sampler_name,
        scheduler,
        shift_video,
        shift_audio,
        use_continuity,
        clips_subdir,
        skip_existing,
    ):
        mp = Path(manifest_path.strip())
        if not mp.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        data = json.loads(mp.read_text(encoding="utf-8"))
        chunks = data["chunks"]

        out_root = Path(folder_paths.get_output_directory())
        clips_dir = out_root / clips_subdir.strip().lstrip("/\\")
        clips_dir.mkdir(parents=True, exist_ok=True)
        continuity_dir = clips_dir / "continuity"
        continuity_dir.mkdir(parents=True, exist_ok=True)

        first_prompt, cont_prompt = _resolve_prompts(prompt, continuity_prompt)

        # Apply H3 sigma shift once
        shifted = _unpack(MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio)))[0]
        sampler = comfy.samplers.sampler_object(sampler_name)
        sigmas = _unpack(BasicScheduler.execute(shifted, scheduler, int(steps), 1.0))[0]
        decoder = VAEDecode()
        saver = SaveVideoFramesNoAudio()

        continuity = None
        saved = 0
        pbar = comfy.utils.ProgressBar(len(chunks))

        for entry in chunks:
            idx = int(entry["index"])
            clip_name = entry["clip_name"]
            out_clip = clips_dir / clip_name
            duration = float(entry["duration_sec"])
            length = int(entry.get("frame_length") or h3_frame_length(duration))
            still_path = continuity_dir / f"last_{idx:03d}.png"

            if skip_existing and out_clip.is_file():
                print(f"[H3 batch] skip existing {clip_name}")
                if use_continuity:
                    continuity = _load_last_frame_from_video(out_clip)
                    if continuity is not None:
                        _save_continuity_still(continuity, still_path)
                        print(f"[H3 batch] continuity frame from skipped {clip_name}")
                saved += 1
                pbar.update(1)
                continue

            audio = _load_audio(Path(entry["path"]))

            # Always keep character ref as <Picture 1>.
            # When continuing, add previous last frame as <Picture 2>.
            ref_images = {"ref_image_1": ref_image}
            if use_continuity and continuity is not None:
                ref_images["ref_image_2"] = continuity
                use_prompt = cont_prompt
                mode = "continuity (<Picture 1> identity + <Picture 2> last frame)"
            else:
                use_prompt = first_prompt
                mode = "first shot (<Picture 1> only)"

            print(
                f"[H3 batch] clip {idx + 1}/{len(chunks)}: {duration:.2f}s -> "
                f"{length} frames -> {clip_name} [{mode}]"
            )

            r2v = _unpack(
                MiniMaxH3ReferenceToVideo.execute(
                    clip,
                    vae,
                    audio_vae,
                    use_prompt,
                    int(width),
                    int(height),
                    int(length),
                    ref_image_size="match",
                    ref_images=ref_images,
                    ref_videos=None,
                    ref_video_audios=None,
                    ref_audios={"ref_audio_1": audio},
                )
            )
            positive, latent = r2v[0], r2v[1]

            guider = Guider_Basic(shifted)
            guider.set_conds(positive)

            noise = Noise_RandomNoise(int(rng) + idx)
            latent_work = latent.copy()
            latent_image = latent_work["samples"]
            latent_image = comfy.sample.fix_empty_latent_channels(
                guider.model_patcher,
                latent_image,
                latent_work.get("downscale_ratio_spacial", None),
                latent_work.get("downscale_ratio_temporal", None),
            )
            latent_work["samples"] = latent_image

            x0_output = {}
            callback = latent_preview.prepare_callback(
                guider.model_patcher, sigmas.shape[-1] - 1, x0_output
            )
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            samples = guider.sample(
                noise.generate_noise(latent_work),
                latent_image,
                sampler,
                sigmas,
                denoise_mask=None,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=noise.seed,
            )
            samples = samples.to(comfy.model_management.intermediate_device())
            out_latent = latent_work.copy()
            out_latent.pop("downscale_ratio_spacial", None)
            out_latent.pop("downscale_ratio_temporal", None)
            out_latent["samples"] = samples

            images = decoder.decode(vae, out_latent)[0]
            saver.save(
                images,
                filename_prefix=str(Path(clips_subdir) / "clip"),
                fps=24.0,
                exact_filename=clip_name,
            )
            if not out_clip.is_file():
                matches = list(clips_dir.rglob(clip_name))
                if matches:
                    if matches[0].resolve() != out_clip.resolve():
                        out_clip.write_bytes(matches[0].read_bytes())
            if not out_clip.is_file():
                raise FileNotFoundError(f"Failed to write clip: {out_clip}")

            if use_continuity:
                continuity = images[-1:].clone()
                _save_continuity_still(continuity, still_path)
            saved += 1
            pbar.update(1)

        return {
            "ui": {
                "text": [
                    f"Saved {saved} clips -> {clips_dir}"
                    + (" (scene continuity ON)" if use_continuity else "")
                ]
            },
            "result": (str(clips_dir), str(mp), saved),
        }


class AceH3FullMusicVideo:
    """
    One-shot: ACE API → chunk → H3 Ref2VA batch → assemble final music_video.mp4
    (Convenience wrapper; also available as separate nodes in the full graph.)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "ref_image": ("IMAGE",),
                "tags": ("STRING", {"multiline": True, "default": "cinematic indie rock, anthemic chorus"}),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "duration_sec": ("FLOAT", {"default": 20.0, "min": 5.0, "max": 360.0, "step": 1.0}),
                "chunk_seconds": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 15.0, "step": 0.1}),
                "prompt": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT}),
                "continuity_prompt": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT_CONTINUITY}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 432, "min": 64, "max": 4096, "step": 32}),
                "steps": ("INT", {"default": 15, "min": 1, "max": 50, "tooltip": "15 for quality; 4 is fastest with Turbo LoRA."}),
                "rng": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "use_continuity": ("BOOLEAN", {"default": True}),
                "ace_api_url": ("STRING", {"default": "http://127.0.0.1:8001"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_path",)
    FUNCTION = "run"
    CATEGORY = "video/music_video"
    OUTPUT_NODE = True

    def run(
        self,
        model,
        clip,
        vae,
        audio_vae,
        ref_image,
        tags,
        lyrics,
        duration_sec,
        chunk_seconds,
        prompt,
        continuity_prompt,
        width,
        height,
        steps,
        rng,
        use_continuity,
        ace_api_url,
    ):
        from .nodes_ace_api import generate_ace_track
        from .nodes_audio import ChunkAudioForMiniMaxH3

        path, audio, _dur = generate_ace_track(
            tags=tags,
            lyrics=lyrics,
            duration_sec=float(duration_sec),
            api_url=ace_api_url,
            relative_out="audio/track.wav",
        )
        chunk_out = ChunkAudioForMiniMaxH3().chunk(
            audio,
            float(chunk_seconds),
            "audio/chunks",
            "manifest.json",
            "audio/track.wav",
            True,
        )
        manifest_path = chunk_out["result"][0] if isinstance(chunk_out, dict) else chunk_out[0]

        batch = MiniMaxH3MusicVideoBatch().run(
            model,
            clip,
            vae,
            audio_vae,
            ref_image,
            manifest_path,
            prompt,
            continuity_prompt,
            int(width),
            int(height),
            int(steps),
            int(rng),
            "res_multistep",
            "simple",
            12.0,
            3.0,
            bool(use_continuity),
            "music_video/clips",
            False,
        )
        clips_dir = batch["result"][0] if isinstance(batch, dict) else batch[0]

        final = AssembleMusicVideoFFmpeg().assemble(
            manifest_path,
            clips_dir,
            "music_video/music_video.mp4",
            24.0,
            "",
        )
        final_path = final["result"][0] if isinstance(final, dict) else final[0]
        return {
            "ui": {"text": [f"Final: {final_path} (ACE track muxed, H3 audio discarded)"]},
            "result": (final_path,),
        }
