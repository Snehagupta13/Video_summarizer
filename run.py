import sys
import os
import json
from state import VideoState
from edges import app

def validate_video_file(video_path: str) -> None:
    """Validate input video file"""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"File not found: {video_path}")
    if not video_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise ValueError("Unsupported video format")

def main():
    try:
        if len(sys.argv) != 2:
            raise SystemExit("Usage: python run.py <video_path>")

        input_path = sys.argv[1]
        validate_video_file(input_path)

        initial_state = VideoState(input_video=input_path)

        print("🚀 Starting full pipeline...")
        final_output = app.invoke(initial_state)

        validated = VideoState(**final_output)
        print(validated.model_dump_json(indent=2))

        if validated.output_files:
            print("\n📂 Output files:")
            for t, path in validated.output_files.items():
                print(f"- {t}: {os.path.abspath(path)}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()