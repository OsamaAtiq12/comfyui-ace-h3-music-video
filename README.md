# ACE-Step → MiniMax H3 music video

Custom nodes for chunking ACE-Step audio, saving silent H3 clips, and assembling a final music video muxed with the **original ACE-Step track** (H3 per-clip audio is discarded).

## Nodes

| Node | Role |
| --- | --- |
| `AceStep15GenerateViaAPI` | Optional: bodycamhooker ACE API (:8001). Prefer native Comfy ACE nodes + `scripts/download_ace_step15.ps1` |
| `SaveAudioTrackWav` | Write full song WAV under `output/` |
| `ChunkAudioForMiniMaxH3` | Split into 2–15s windows (default **12s**) + `manifest.json` |
| `LoadAudioFromPath` | Load chunk WAV by absolute/relative path |
| `LoadManifestAudioChunk` | Load chunk N from manifest |
| `SecondsToH3FrameLength` | Snap seconds → H3 `17k+5` frame length (cap 362) |
| `ExtractLastFrame` | Continuity helper |
| `SaveVideoFramesNoAudio` | Encode IMAGE → MP4 **without audio** |
| `AssembleMusicVideoFFmpeg` | Concat clips + mux ACE `track.wav` |

## Hard limits (MiniMax H3)

- Reference audio **2–15s** per call; total ref audio ≤ **15s**
- Audio **cannot** be used without ≥1 reference **image or video**
- Output video ≤ **~15s** (frame length ≤ **362** @ 24fps)

## Orchestrator

ComfyUI does not loop well over N chunks. Use:

```powershell
python scripts\run_ace_h3_music_video.py --generate-music --ref-image input\my_style.png --chunk-seconds 12
```

See project `README.md` → Music video pipeline.
