"""ACE-Step → MiniMax H3 R2V music-video helpers (chunk / assemble / path I/O)."""

WEB_DIRECTORY = "./web"

from .nodes_ace_api import AceStep15GenerateViaAPI
from .nodes_audio import (
    SaveAudioTrackWav,
    ChunkAudioForMiniMaxH3,
    LoadAudioFromPath,
    LoadManifestAudioChunk,
    SecondsToH3FrameLength,
    TrimTrailingSilence,
)
from .nodes_batch import MiniMaxH3MusicVideoBatch, AceH3FullMusicVideo
from .nodes_duration import EstimateSongDurationFromLyrics
from .nodes_video import (
    ExtractLastFrame,
    SaveVideoFramesNoAudio,
    AssembleMusicVideoFFmpeg,
    PreviewFinalMusicVideo,
)

NODE_CLASS_MAPPINGS = {
    "AceStep15GenerateViaAPI": AceStep15GenerateViaAPI,
    "SaveAudioTrackWav": SaveAudioTrackWav,
    "ChunkAudioForMiniMaxH3": ChunkAudioForMiniMaxH3,
    "LoadAudioFromPath": LoadAudioFromPath,
    "LoadManifestAudioChunk": LoadManifestAudioChunk,
    "SecondsToH3FrameLength": SecondsToH3FrameLength,
    "TrimTrailingSilence": TrimTrailingSilence,
    "ExtractLastFrame": ExtractLastFrame,
    "SaveVideoFramesNoAudio": SaveVideoFramesNoAudio,
    "AssembleMusicVideoFFmpeg": AssembleMusicVideoFFmpeg,
    "PreviewFinalMusicVideo": PreviewFinalMusicVideo,
    "MiniMaxH3MusicVideoBatch": MiniMaxH3MusicVideoBatch,
    "AceH3FullMusicVideo": AceH3FullMusicVideo,
    "EstimateSongDurationFromLyrics": EstimateSongDurationFromLyrics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AceStep15GenerateViaAPI": "ACE-Step 1.5 Generate (API)",
    "SaveAudioTrackWav": "Save Audio Track (WAV)",
    "ChunkAudioForMiniMaxH3": "Chunk Audio for MiniMax H3",
    "LoadAudioFromPath": "Load Audio From Path",
    "LoadManifestAudioChunk": "Load Manifest Audio Chunk",
    "SecondsToH3FrameLength": "Seconds → H3 Frame Length",
    "TrimTrailingSilence": "Trim Trailing Silence",
    "ExtractLastFrame": "Extract Last Frame",
    "SaveVideoFramesNoAudio": "Save Video Frames (No Audio)",
    "AssembleMusicVideoFFmpeg": "Assemble Music Video (FFmpeg)",
    "PreviewFinalMusicVideo": "Preview Final Music Video",
    "MiniMaxH3MusicVideoBatch": "MiniMax H3 Music Video Batch (all chunks)",
    "AceH3FullMusicVideo": "ACE→H3 Full Music Video (one-shot)",
    "EstimateSongDurationFromLyrics": "Estimate Song Duration From Lyrics",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
