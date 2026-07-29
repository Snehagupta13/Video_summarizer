import os
import sys
import types
import importlib.machinery
import torch
import imageio_ffmpeg

def _neutralize_torchcodec():
    """Pre-empt torchcodec's shared-library RuntimeError (missing system ffmpeg
    .so files) so unrelated optional imports elsewhere (e.g. sentence_transformers,
    datasets) see it as simply unavailable instead of crashing the whole process."""
    if "torchcodec" in sys.modules:
        return
    try:
        import torchcodec  # noqa: F401
    except RuntimeError:
        stub = types.ModuleType("torchcodec")
        stub.__spec__ = importlib.machinery.ModuleSpec("torchcodec", loader=None)
        stub.decoders = types.ModuleType("torchcodec.decoders")
        stub.decoders.__spec__ = importlib.machinery.ModuleSpec("torchcodec.decoders", loader=None)
        stub.decoders.AudioDecoder = None
        stub.decoders.VideoDecoder = None
        sys.modules["torchcodec"] = stub
        sys.modules["torchcodec.decoders"] = stub.decoders
    except ImportError:
        pass

_neutralize_torchcodec()

# Make whisperx's ffmpeg subprocess calls work without a system/apt ffmpeg install.
# imageio_ffmpeg bundles a versioned binary (e.g. "ffmpeg-linux-x86_64-v7.0.2"), but
# whisperx.load_audio shells out to the literal name "ffmpeg", so symlink it.
_ffmpeg_bin_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".ffmpeg_bin")
os.makedirs(_ffmpeg_bin_dir, exist_ok=True)
_ffmpeg_link = os.path.join(_ffmpeg_bin_dir, "ffmpeg")
if not os.path.exists(_ffmpeg_link):
    os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), _ffmpeg_link)
os.environ["PATH"] = _ffmpeg_bin_dir + os.pathsep + os.environ.get("PATH", "")

import whisperx

device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
model = whisperx.load_model("large-v3", device=device, compute_type=compute_type)

def transcribe_audio(video_path: str) -> str:
    """Transcribe speech from a video's audio track using WhisperX."""
    try:
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        audio = whisperx.load_audio(video_path)
        result = model.transcribe(audio, batch_size=16)

        return " ".join(seg["text"].strip() for seg in result.get("segments", []))
    except Exception as e:
        print(f"[❌] Speech-to-text error: {e}")
        return ""
