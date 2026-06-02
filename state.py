from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional, List, Tuple, Dict, Any
from pathlib import Path

class VideoState(BaseModel):
    input_video: str = Field(..., description="Path to the input video file")
    converted_video: Optional[str] = Field(
        None, 
        description="Path to the converted video file"
    )
    frames: List[str] = Field(
        default_factory=list,
        description="List of paths to extracted frames"
    )
    captions: List[Tuple[str, str]] = Field(
        default_factory=list,
        description="List of (frame_path, caption) tuples"
    )
    final_video: Optional[str] = Field(
        None,
        description="Path to the final processed video"
    )
    extracted_text: str = Field(
        default="",
        description="Text extracted from frames using OCR"
    )
    scene_summary: str = Field(
        default="",
        description="LLM-generated summary of video scenes"
    )
    text_summary: str = Field(
        default="",
        description="Summary of extracted text"
    )
    final_summary: str = Field(
        default="",
        description="Final combined summary"
    )
    
    output_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Dictionary of output file paths by type"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={tuple: lambda v: str(v)},
        validate_default=True,
        extra="forbid"
    )

    @field_validator('input_video', 'converted_video', 'final_video')
    def validate_video_paths(cls, v):
        if v is not None and not Path(v).exists():
            raise ValueError(f"Video file does not exist: {v}")
        return v

    @field_validator('frames', mode='before')
    def validate_frames(cls, v):
        if v is None:
            return []
        return [str(frame) for frame in v]

    @field_validator('captions', mode='before')
    def validate_captions(cls, v):
        if v is None:
            return []
        return [(str(frame), str(caption)) for frame, caption in v]