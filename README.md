# Video Analysis with KoboldCPP

This tool analyzes videos by extracting frames and processing them through a KoboldCPP API endpoint. It generates frame-by-frame analyses and creates progressive summaries, culminating in a comprehensive final summary of the video content.

## Features

- Extracts frames from videos while preserving aspect ratio
- Processes frames in batches to manage memory usage
- Generates rolling summaries that adapt to context length
- Produces a detailed final summary incorporating temporal progression
- Saves all analyses in a structured JSON format

## Installation

### Windows

1. Clone or zip this repository and extract it
2. Download koboldcpp from the lostruins repo and name it koboldcpp.exe and place it in the directory
3. Run kcppvideo-run.bat
4. Wait for the env to setup
5. The GUI will appear. **IMPORTANT: Wait for the model and projector to download and for the KoboldCPP windows to say Ready before running the analysis.**

### No Installation

1. Clone or zip this repository and extract it
2. Download koboldcpp from the lostruins repo and name it koboldcpp.exe and place it in the directory
3. Open KoboldCPP and load the model and image projector

## Usage

Basic usage:
```bash
python kcppvideo.py video_file.mp4
```

Advanced options:
```bash
python kcppvideo.py video_file.mp4 \
    --api-url http://localhost:5001 \
    --template-dir ./templates \
    --max-frames 24 \
    --output-dir ./analysis_output \
    --batch-size 4
```

### Parameters

- `video`: Path to the video file to analyze
- `--api-url`: KoboldCPP API URL (default: http://localhost:5001)
- `--template-dir`: Path to instruction templates (default: ./templates)
- `--max-frames`: Maximum frames to analyze (default: 24)
- `--output-dir`: Output directory (default: video_name_analysis)
- `--batch-size`: Frames per batch (default: 4)

## Output

The tool creates a directory containing:
- `analysis.json`: Complete analysis including:
  - Frame-by-frame analyses
  - Progressive summaries
  - Final comprehensive summary
  - Metadata about the analysis

## Dependencies

See requirements.txt for the complete list of Python dependencies.

## Notes

- The tool automatically manages memory usage by processing frames in batches
- Summary length is dynamically adjusted based on the model's context window
- Frames are automatically resized to a suitable width while preserving aspect ratio
