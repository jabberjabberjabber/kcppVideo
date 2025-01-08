import argparse
import base64
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import io
from kcppapi import KoboldAPI
from kcppwrapper import InstructTemplate
import os
import subprocess
import tempfile
import glob


MAX_NUM_FRAMES = 64

def get_scene_frames(video_path, threshold=0.1, min_gap=4, format='png'):
    """Gets scene change frames with frame numbers and timecode.
    
    Uses ffmpeg's scene detection with a minimum time gap and adds
    machine-readable frame number and SMPTE timecode overlay. Images are 
    resized to max 320px on longest side while maintaining aspect ratio.
    
    Args:
        video_path: Path to input video
        threshold: Scene detection threshold (0.0-1.0) 
        min_gap: Minimum seconds between detected scenes
        format: Output image format ('jpeg' or 'png')
        
    Returns:
        list: Base64 encoded images for all detected scenes
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_pattern = os.path.join(tmpdir, f'scene_%04d.{format}')
        filter_expr = (
            f"select='if(gt(scene,{threshold}),scene,0)"
            f"+if(isnan(prev_selected_t),0,gte(t-prev_selected_t,{min_gap}))"
            f"*between(t-prev_selected_t,-{min_gap},{min_gap})',"
            "setpts=N/FRAME_RATE/TB,"
            "scale='if(gt(iw,ih),min(320,iw),-1):if(gt(iw,ih),-1,min(320,ih))',"  # Resize
            f"drawtext=fontfile='arial.ttf'"
            ":text='FRAME\\:%{n} TIME\\:%{pts\\:hms} TIMECODE\\:%{pts\\:hms\\:24}'"
            ":x=10:y=10:fontcolor=white:fontsize=24:box=1:boxcolor=black"  # Reduced font size for smaller image
        )
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', filter_expr,
            '-vsync', 'vfr',
            output_pattern
        ]
        subprocess.run(cmd, check=True)
        images = []
        for frame_path in sorted(glob.glob(os.path.join(tmpdir, f'scene_*.{format}'))):
            with open(frame_path, 'rb') as f:
                images.append(base64.b64encode(f.read()).decode())
        return images

def calculate_summary_tokens(client: KoboldAPI, current_batch: int, 
                           total_batches: int, current_summary_tokens: int) -> int:
    """ Figure out how many tokens we use for the next summary. """
    max_context = client.get_max_context_length()
    remaining_batches = total_batches - current_batch
    target_summary = int((max_context * 0.7) / remaining_batches)
    input_tokens = (current_summary_tokens + 
                   200 +  
                   100)
    available = max_context - input_tokens
    return min(target_summary, available // 2)

def analyze_video(video_path: str, api_url: str, template_dir: str,
                 max_frames: int = 84, output_dir: str = None,
                 batch_size: int = 4) -> dict:
    """ Analyze an entire video by sending batches of frames to a 
        Koboldcpp API and get a rolling summary.

        Args:
            video_path: Path to video file
            api_url: URL of Kobold API endpoint
            template_dir: Directory containing prompt templates
            max_frames: Maximum number of frames to process
            output_dir: Directory to save results (defaults to video location)
            batch_size: Number of frames to process in each batch

        Returns:
            dict: Contains frame analysis, progressive summaries, and final summary
    """
    client = KoboldAPI(api_url)
    wrapper = InstructTemplate(template_dir, api_url)
    max_context = client.get_max_context_length()
    video_path = Path(video_path)
    out_path = Path(output_dir) if output_dir else video_path.parent / f"{video_path.stem}_analysis"
    out_path.mkdir(exist_ok=True)
    results = {
        "analysis": [],
        "progressive_summaries": [],
        "final_summary": None,
        "metadata": {
            "video_path": str(video_path.absolute()),
            "api_url": api_url,
            "max_frames": max_frames,
            "batch_size": batch_size
        }
    }
    print(f"Extracting frames from {video_path}...")
    frames = get_scene_frames(video_path)
    results["metadata"]["frame_count"] = len(frames)
    frame_batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    total_batches = len(frame_batches)
    current_summary = None
    current_summary_tokens = 0

    for batch_idx, frame_batch in enumerate(frame_batches):
        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        batch_analysis = analyze_frame_batch(
            client=client,
            wrapper=wrapper,
            frame_batch=frame_batch,
            batch_idx=batch_idx,
            batch_size=batch_size
        )
        results["analysis"].append({
            "batch": batch_idx + 1,
            "frame_range": f"{batch_idx * batch_size + 1}-{min((batch_idx + 1) * batch_size, len(frames))}",
            "analysis": batch_analysis
        })
        if current_summary is None:
            current_summary = batch_analysis
            current_summary_tokens = client.count_tokens(current_summary)["count"]
        else:
            continue
            new_summary = update_progressive_summary(
                client=client,
                wrapper=wrapper,
                current_summary=current_summary,
                batch_analysis=batch_analysis,
                batch_idx=batch_idx,
                total_batches=total_batches,
                current_summary_tokens=current_summary_tokens
            )
            
            results["progressive_summaries"].append({
                "batch": batch_idx + 1,
                "frame_range": f"{1}-{min((batch_idx + 1) * batch_size, len(frames))}",
                "summary": new_summary
            })
            current_summary = new_summary
            current_summary_tokens = client.count_tokens(current_summary)["count"]
    if frame_batches:
        final_summary = generate_final_summary(
            client=client,
            wrapper=wrapper,
            results=results
        )
        results["final_summary"] = final_summary
    results_file = out_path / "analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return results

def analyze_frame_batch(client, wrapper, frame_batch, batch_idx, batch_size):
    """ Analyze a batch of frames using the Kobold API.
    """
    frame_instruction = "Describe the objects and actions in these frames."
    frame_system = "You are an expert video analyzer."
    frame_prompt = wrapper.wrap_prompt(frame_instruction, "", frame_system)[0]
    return client.generate(
        prompt=frame_prompt,
        images=frame_batch,
        temperature=0.1,
        max_length=200,
        top_p=1,
        top_k=100,
        rep_pen=1
    )

def update_progressive_summary(client, wrapper, current_summary, batch_analysis,
                             batch_idx, total_batches, current_summary_tokens):
    """ Update the running summary with new batch analysis.
    """
    max_generation = calculate_summary_tokens(
        client,
        batch_idx,
        total_batches,
        current_summary_tokens
    )
    summary_instruction = (
        "Update the summary, incorporating the previous summary with the new events."
    )
    summary_content = f"\nCurrent summary: {current_summary}\nNew events: {batch_analysis}\n"
    summary_system = "You are summarizing a video sequence progressively."
    summary_prompt = wrapper.wrap_prompt(
        summary_instruction,
        summary_content,
        summary_system
    )[1]   
    return client.generate(
        prompt=summary_prompt,
        temperature=0,
        max_length=500,
        top_p=1,
        top_k=100,
        rep_pen=1
    )

def generate_final_summary(client, wrapper, results):
    """ Generate comprehensive final summary using all analyses.
    """
    max_context = client.get_max_context_length()
    
    all_analyses = "\n\n".join(
        f"Frames {analysis['frame_range']}: {analysis['analysis']}" 
        for analysis in results["analysis"]
    )
    final_instruction = (
        "Describe the objects and actions in these frames. Summarize the events as if giving a narrative."
    )
    final_content = f"\nAnalyses: {all_analyses}\n"
    final_system = "You are an expert video analyzer. You use plain language."
    
    final_prompt = wrapper.wrap_prompt(
        final_instruction,
        final_content,
        final_system
    )[1]
    prompt_tokens = client.count_tokens(final_prompt)["count"]
    max_generation = (max_context - prompt_tokens) // 2
    
    return client.generate(
        prompt=final_prompt,
        temperature=0,
        max_length=max_generation,
        top_p=1
    )

def main():
    parser = argparse.ArgumentParser(description="Analyze video using KoboldCPP")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--api-url", default="http://localhost:5001",
                      help="KoboldCPP API URL (default: http://localhost:5001)")
    parser.add_argument("--template-dir", default="./templates",
                      help="Path to instruction templates (default: ./templates)")
    parser.add_argument("--max-frames", type=int, default=24,
                      help="Maximum frames to analyze (default: 24)")
    parser.add_argument("--output-dir",
                      help="Output directory (default: video_name_analysis)")
    parser.add_argument("--batch-size", type=int, default=1,
                      help="Frames per batch (default: 2)")
    args = parser.parse_args()
    try:
        results = analyze_video(
            args.video,
            args.api_url,
            args.template_dir,
            args.max_frames,
            args.output_dir,
            args.batch_size
        )
        print("\nVideo Summary:")
        print("-" * 80)
        print(results["final_summary"])
        print("\nSaved to output directory.")
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())