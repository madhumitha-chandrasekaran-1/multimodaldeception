"""
01_extract_audio.py
--------------------
Extracts audio from all .mp4 clips in data/truth/ and data/lie/
and saves them as 16kHz mono .wav files in output/audio/truth/ and output/audio/lie/

Usage:
    python 01_extract_audio.py --data_dir data/ --output_dir output/audio/
"""

import os
import subprocess
import argparse
from pathlib import Path


def extract_audio(video_path: Path, output_path: Path, sample_rate: int = 16000) -> bool:
    """
    Extract audio from a video file using ffmpeg.
    Converts to mono, 16kHz WAV — standard for speech processing.

    Args:
        video_path:   Path to input .mp4 file
        output_path:  Path to output .wav file
        sample_rate:  Target sample rate (default 16000 Hz)

    Returns:
        True if successful, False otherwise
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),     # input file
        "-vn",                      # no video
        "-acodec", "pcm_s16le",    # 16-bit PCM WAV
        "-ar", str(sample_rate),   # sample rate
        "-ac", "1",                 # mono
        "-y",                       # overwrite output if exists
        str(output_path)
    ]

    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def process_folder(input_folder: Path, output_folder: Path, label: str) -> dict:
    """
    Process all .mp4 files in a folder and extract their audio.

    Returns a summary dict with counts.
    """
    video_files = sorted(input_folder.glob("*.mp4"))

    if not video_files:
        print(f"  ⚠️  No .mp4 files found in {input_folder}")
        return {"total": 0, "success": 0, "failed": 0}

    success, failed = 0, 0

    for video_path in video_files:
        output_path = output_folder / (video_path.stem + ".wav")

        if output_path.exists():
            print(f"  ⏭️  Skipping (already exists): {video_path.name}")
            success += 1
            continue

        ok = extract_audio(video_path, output_path)

        if ok:
            print(f"  ✅  {video_path.name} → {output_path.name}")
            success += 1
        else:
            print(f"  ❌  Failed: {video_path.name}")
            failed += 1

    return {"total": len(video_files), "success": success, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Extract audio from deception detection video clips")
    parser.add_argument("--data_dir",   type=str, default="data/",        help="Root data directory containing truth/ and lie/ subfolders")
    parser.add_argument("--output_dir", type=str, default="output/audio/", help="Output directory for extracted .wav files")
    parser.add_argument("--sample_rate", type=int, default=16000,          help="Target audio sample rate (default: 16000 Hz)")
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Validate input structure
    truth_dir = data_dir / "truthful"
    lie_dir   = data_dir / "deceptive"

    if not truth_dir.exists():
        raise FileNotFoundError(f"Truth folder not found: {truth_dir}")
    if not lie_dir.exists():
        raise FileNotFoundError(f"Lie folder not found: {lie_dir}")

    print("=" * 55)
    print("  Deception Detection — Audio Extraction")
    print("=" * 55)
    print(f"  Data dir:    {data_dir.resolve()}")
    print(f"  Output dir:  {output_dir.resolve()}")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print("=" * 55)

    # Process truth clips
    print(f"\n📁 Processing TRUTH clips...")
    truth_stats = process_folder(
        input_folder  = truth_dir,
        output_folder = output_dir / "truthful",
        label         = "truthful"
    )

    # Process lie clips
    print(f"\n📁 Processing LIE clips...")
    lie_stats = process_folder(
        input_folder  = lie_dir,
        output_folder = output_dir / "deceptive",
        label         = "deceptive"
    )

    # Summary
    print("\n" + "=" * 55)
    print("  Summary")
    print("=" * 55)
    print(f"  Truth clips:  {truth_stats['success']}/{truth_stats['total']} extracted successfully")
    print(f"  Lie clips:    {lie_stats['success']}/{lie_stats['total']} extracted successfully")
    total = truth_stats['total'] + lie_stats['total']
    total_ok = truth_stats['success'] + lie_stats['success']
    print(f"  Total:        {total_ok}/{total} clips processed")

    if truth_stats['failed'] + lie_stats['failed'] > 0:
        print(f"\n  ⚠️  {truth_stats['failed'] + lie_stats['failed']} clips failed — check ffmpeg installation")
    else:
        print(f"\n  🎉 All clips extracted successfully!")
    print("=" * 55)


if __name__ == "__main__":
    main()