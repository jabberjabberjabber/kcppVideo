import argparse
import base64
import json
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
import io
import decord
import numpy as np
from PIL import Image
from kcppapi import KoboldAPI
from kcppwrapper import InstructTemplate

def resize_preserve_aspect(width: int, height: int, target_width: int = 320) -> Tuple[int, int]:
    """ Resize image preserving aspect ratio. """
    ratio = target_width / width
    new_width = target_width
    new_height = int(height * ratio)
    return new_width, new_height

def smart_nframes(total_frames: int, target_frames: int) -> int:
    """ Calculate number of frames to extract. """
    nframes = min(target_frames, total_frames)
    if nframes < 2:
        raise ValueError(f"Need at least 2 frames, got {nframes}")
    return nframes

def extract_frames(video_path: str, max_frames: int) -> List[str]:
    """ Extract and resize frames from video.
    
        Koboldcpp wants images a list of base64 objects.
    """
    
    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)
    nframes = smart_nframes(total_frames, max_frames)
    
    # Figure out where to space the frames to cover the whole video
    idx = np.linspace(0, total_frames - 1, nframes).round().astype(int).tolist()
    
    # We extract frames in batches so it doesn't eat all the memory
    frames = []
    batch_size = 4  # koboldcpp can hand 4 images at a time
    for i in range(0, len(idx), batch_size):
        batch_idx = idx[i:i + batch_size]
        batch = vr.get_batch(batch_idx).asnumpy()
        for frame_array in batch:
            frame_pil = Image.fromarray(frame_array)
            width, height = frame_pil.size    
            new_width, new_height = resize_preserve_aspect(width, height)
            frame_pil = frame_pil.resize((new_width, new_height))
            
            buffered = io.BytesIO()
            frame_pil.save(buffered, format="JPEG")
            frame_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            frames.append(frame_base64)
    return frames

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
                 max_frames: int = 24, output_dir: str = None,
                 batch_size: int = 4) -> dict:
    """ Analyze an entire video by sending batches of frames to a 
        Koboldcpp API and get a rolling summary.
    """
    client = KoboldAPI(api_url)
    wrapper = InstructTemplate(template_dir, api_url)
    max_context = client.get_max_context_length()
    video_path = Path(video_path)
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = video_path.parent / f"{video_path.stem}_analysis"
    out_path.mkdir(exist_ok=True)

    print(f"Extracting frames from {video_path}...")
    frames = extract_frames(video_path, max_frames)
    
    frame_batches = [frames[i:i + batch_size] for i in range(0, len(frames), batch_size)]
    total_batches = len(frame_batches)
    
    results = {
        "frame_analyses": [],
        "progressive_summaries": [],
        "final_summary": None,
        "metadata": {
            "video_path": str(video_path.absolute()),
            "frame_count": len(frames),
            "api_url": api_url
        }
    }
    
    current_summary = None
    current_summary_tokens = 0
    
    for batch_idx, frame_batch in enumerate(frame_batches):
        print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        
        start_frame = batch_idx * batch_size + 1
        frame_instruction = (
            f"Analyze frames {start_frame}-{start_frame + len(frame_batch) - 1}. "
            f"Describe the objects and actions."
        )
        frame_system = "You are a helpful assistant doing a video analysis with a rolling summary."
        frame_prompt = wrapper.wrap_prompt(frame_instruction, "", frame_system)[0]
        batch_analysis = client.generate(
            prompt=frame_prompt,
            images=frame_batch,
            temperature=0,
            max_length=500,
            top_p=1
        )
        results["frame_analyses"].append({
            "frames": list(range(start_frame, start_frame + len(frame_batch))),
            "analysis": batch_analysis
        })
        
        if current_summary is None:
            current_summary = batch_analysis
            current_summary_tokens = client.count_tokens(current_summary)["count"]
        else:
            max_generation = calculate_summary_tokens(
                client,
                batch_idx,
                total_batches,
                current_summary_tokens
            )
            summary_instruction = (
                "Update the summary, incorporating the previous summary with the new events."
                "The current summary needs to be summarized again."
                "The goal is to have one new summary that is a complete summary of everything."
            )
            summary_content = f"\n\nCurrent summary: {current_summary}\nNew events: {batch_analysis}\n\n"
            summary_system = "You are summarizing a video sequence progressively."
            
            summary_prompt = wrapper.wrap_prompt(
                summary_instruction,
                summary_content,
                summary_system
            )[0]
            new_summary = client.generate(
                prompt=summary_prompt,
                temperature=0,
                max_length=max_generation,
                top_p=1
            )
            if (batch_idx + 1) < total_batches:
                results["progressive_summaries"].append({
                    "batch": batch_idx + 1,
                    "summary": new_summary
                })
                current_summary = new_summary
                current_summary_tokens = client.count_tokens(current_summary)["count"]
            else:
                # Store the progressive summary but don't use it as final
                results["progressive_summaries"].append({
                    "batch": batch_idx + 1,
                    "summary": new_summary
                })
                current_summary = new_summary
    
                max_context = client.get_max_context_length()
                
                all_analyses = "\n\n".join(f"Frames {analysis['frames']}: {analysis['analysis']}" 
                                          for analysis in results["frame_analyses"])
                
                final_instruction = (
                    "Create a comprehensive final summary of the entire video sequence. "
                    "Focus on the key events, patterns, and notable changes across all frames. "
                    "Integrate the temporal progression while maintaining clarity and coherence. "
                    "Pay special attention to any significant shifts in activity, lighting, or scene composition."
                )
                final_content = f"\n\nFrame-by-frame analyses:\n{all_analyses}\n\n"
                final_system = "You are creating a thorough final summary of a video sequence."
                
                final_prompt = wrapper.wrap_prompt(
                    final_instruction,
                    final_content,
                    final_system
                )[0]
                
                prompt_tokens = client.count_tokens(final_prompt)["count"]
                max_generation = (max_context - prompt_tokens) // 2
                
                final_summary = client.generate(
                    prompt=final_prompt,
                    temperature=0,
                    max_length=max_generation,
                    top_p=1
                )
                
                results["final_summary"] = final_summary
                
    results_file = out_path / "analysis.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    return results

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
    parser.add_argument("--batch-size", type=int, default=4,
                      help="Frames per batch (default: 4)")
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