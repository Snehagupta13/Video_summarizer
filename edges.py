from langgraph.graph import StateGraph
from nodes import (
    process_video,
    process_captions,
    process_easyocr,
    summarize_captions,
    run_chatbot_node,
)
from state import VideoState

# Create a LangGraph state graph for the video processing pipeline
builder = StateGraph(VideoState)

# Add nodes (each one is a function that returns an updated state or dict)
builder.add_node("process_video", process_video)
builder.add_node("process_captions", process_captions)
builder.add_node("process_easyocr", process_easyocr)
builder.add_node("summarize_captions", summarize_captions)
builder.add_node("run_chatbot_node", run_chatbot_node)

# Set the entry point of the workflow
builder.set_entry_point("process_video")

# Define linear flow between nodes
builder.add_edge("process_video", "process_captions")
builder.add_edge("process_captions", "process_easyocr")
builder.add_edge("process_easyocr", "summarize_captions")
builder.add_edge("summarize_captions", "run_chatbot_node")

# Compile the graph into an executable app
app = builder.compile()
