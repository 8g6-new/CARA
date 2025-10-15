#!/usr/bin/env python3
"""
CARA vs Librosa Beat Tracking Comparison Tool

This script compares CARA's beat tracking output with librosa's
implementation, providing comprehensive analysis and validation.

Usage:
    python compare_cara_librosa_beat.py [cara_output.txt] [audio_file.wav]

If no arguments provided, uses default paths:
    - CARA output: ../outputs/beat_tracking_default.txt
    - Audio file: files/riad.wav
"""

import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

def load_cara_beats(cara_file):
    """
    Load CARA beat-tracking output from a text file and return a structured dictionary of parameters, results, and beat arrays.
    
    Parameters:
        cara_file (str): Path to a CARA beat output text file. The file may contain header lines beginning with '#' (key:value pairs) and beat lines with "index frame time".
    
    Returns:
        dict or None: On success, a dictionary with keys:
            - 'params': dict of parsed CARA parameters (e.g., 'tightness', 'trim', 'sparse').
            - 'results': dict of parsed result fields (e.g., 'num_beats', 'tempo_bpm', 'confidence', 'frame_rate', 'total_frames').
            - 'beat_times': numpy.ndarray of beat times in seconds.
            - 'beat_frames': numpy.ndarray of beat frame indices.
            - 'filename': the input file path.
        Returns None if the file cannot be read or parsing fails.
    """
    try:
        # Parse the header to extract parameters and results
        params = {}
        results = {}
        beat_data = []
        
        with open(cara_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    # Parse parameter and result lines
                    if 'tightness:' in line:
                        params['tightness'] = float(line.split(':')[1].strip())
                    elif 'trim:' in line:
                        params['trim'] = 'true' in line.lower()
                    elif 'sparse:' in line:
                        params['sparse'] = 'true' in line.lower()
                    elif 'num_beats:' in line:
                        results['num_beats'] = int(line.split(':')[1].strip())
                    elif 'tempo_bpm:' in line:
                        results['tempo_bpm'] = float(line.split(':')[1].strip())
                    elif 'confidence:' in line:
                        results['confidence'] = float(line.split(':')[1].strip())
                    elif 'frame_rate:' in line:
                        results['frame_rate'] = float(line.split(':')[1].split()[0])
                    elif 'total_frames:' in line:
                        results['total_frames'] = int(line.split(':')[1].strip())
                elif line and not line.startswith('#'):
                    # Parse beat data: beat_index frame_position time_seconds
                    parts = line.split()
                    if len(parts) == 3:
                        beat_data.append({
                            'index': int(parts[0]),
                            'frame': int(parts[1]),
                            'time': float(parts[2])
                        })
        
        # Extract arrays
        beat_times = np.array([b['time'] for b in beat_data])
        beat_frames = np.array([b['frame'] for b in beat_data])
        
        return {
            'params': params,
            'results': results,
            'beat_times': beat_times,
            'beat_frames': beat_frames,
            'filename': cara_file
        }
        
    except Exception as e:
        print(f"❌ Error loading CARA beats from {cara_file}: {e}")
        return None

def compute_librosa_beats(audio_file, cara_params=None):
    """
    Compute beat times and frames for an audio file using librosa, attempting to match provided CARA tracking parameters.
    
    Parameters:
        audio_file (str): Path to the audio file to analyze.
        cara_params (dict, optional): CARA parameters to influence librosa settings; when present, `tightness` and `trim` are read and applied.
    
    Returns:
        dict: A results dictionary with keys:
            - 'tempo_bpm' (float): Estimated tempo in beats per minute.
            - 'beat_times' (numpy.ndarray): Beat times in seconds.
            - 'beat_frames' (numpy.ndarray): Beat frame indices (integers).
            - 'num_beats' (int): Number of detected beats.
            - 'params' (dict): Parameters used for computation ('tightness', 'trim', 'sr', 'hop_length').
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        print(f"📁 Loaded audio: {len(y)} samples, {sr} Hz, {len(y)/sr:.3f}s")
        
        # STFT parameters (matching CARA's likely configuration)
        hop_length = 512
        
        print(f"🔧 Beat tracking parameters: hop_length={hop_length}")
        
        # Set beat tracking parameters
        if cara_params:
            tightness = cara_params.get('tightness', 100.0)
            trim = cara_params.get('trim', True)
        else:
            tightness = 100.0
            trim = True
        
        # Compute beats with matching parameters
        tempo, beats = librosa.beat.beat_track(
            y=y, 
            sr=sr,
            hop_length=hop_length,
            tightness=tightness,
            trim=trim,
            units='time'
        )
        
        # Also get frame-based beats for comparison
        _, beats_frames = librosa.beat.beat_track(
            y=y, 
            sr=sr,
            hop_length=hop_length,
            tightness=tightness,
            trim=trim,
            units='frames'
        )
        
        tempo_bpm = float(tempo[0])
        print(f"🎯 Librosa beats: {len(beats)} beats, tempo: {tempo_bpm:.2f} BPM")
        
        return {
            'tempo_bpm': tempo_bpm,
            'beat_times': beats,
            'beat_frames': beats_frames.astype(int),
            'num_beats': len(beats),
            'params': {
                'tightness': tightness,
                'trim': trim,
                'sr': sr,
                'hop_length': hop_length
            }
        }
        
    except Exception as e:
        print(f"❌ Error computing librosa beats from {audio_file}: {e}")
        return None

def analyze_beat_comparison(cara_data, librosa_data):
    """
    Compute comparison metrics between CARA and Librosa beat-tracking results.
    
    Parameters:
        cara_data (dict): CARA result dictionary containing at least 'beat_times' (array-like of times in seconds)
            and 'results' with 'tempo_bpm'.
        librosa_data (dict): Librosa result dictionary containing at least 'beat_times' (array-like of times in seconds)
            and 'tempo_bpm'.
    
    Returns:
        dict: Metrics describing tempo agreement and beat alignment with these keys:
            - 'cara_tempo': CARA tempo in BPM.
            - 'librosa_tempo': Librosa tempo in BPM.
            - 'tempo_diff': Absolute tempo difference (BPM).
            - 'tempo_rel_diff': Relative tempo difference as percentage of Librosa tempo.
            - 'cara_num_beats': Number of CARA beats.
            - 'librosa_num_beats': Number of Librosa beats.
            - 'beat_alignment_score': Matched beats divided by the larger of the two beat counts.
            - 'mean_beat_error': Mean absolute timing error for matched beats (seconds), or infinity if none matched.
            - 'matched_beats': Count of CARA beats with a Librosa beat within the tolerance.
            - 'precision': matched_beats / cara_num_beats.
            - 'recall': matched_beats / librosa_num_beats.
            - 'f1_score': Harmonic mean of precision and recall.
            - 'tolerance': Time tolerance (seconds) used to consider a beat as matched (0.07).
    """
    cara_times = cara_data['beat_times']
    librosa_times = librosa_data['beat_times']
    cara_tempo = cara_data['results']['tempo_bpm']
    librosa_tempo = librosa_data['tempo_bpm']
    
    # Tempo comparison
    tempo_diff = abs(cara_tempo - librosa_tempo)
    tempo_rel_diff = tempo_diff / librosa_tempo * 100 if librosa_tempo > 0 else float('inf')
    
    # Beat alignment analysis
    if len(cara_times) == 0 or len(librosa_times) == 0:
        return {
            'cara_tempo': cara_tempo,
            'librosa_tempo': librosa_tempo,
            'tempo_diff': tempo_diff,
            'tempo_rel_diff': tempo_rel_diff,
            'cara_num_beats': len(cara_times),
            'librosa_num_beats': len(librosa_times),
            'beat_alignment_score': 0.0,
            'mean_beat_error': float('inf'),
            'matched_beats': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # Find closest matches between beat sequences
    tolerance = 0.07  # 70ms tolerance (typical for beat tracking evaluation)
    matched_beats = 0
    beat_errors = []
    
    for cara_beat in cara_times:
        # Find closest librosa beat
        diffs = np.abs(librosa_times - cara_beat)
        min_diff = np.min(diffs)
        
        if min_diff <= tolerance:
            matched_beats += 1
            beat_errors.append(min_diff)
    
    # Compute metrics
    precision = matched_beats / len(cara_times) if len(cara_times) > 0 else 0.0
    recall = matched_beats / len(librosa_times) if len(librosa_times) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mean_beat_error = np.mean(beat_errors) if beat_errors else float('inf')
    beat_alignment_score = matched_beats / max(len(cara_times), len(librosa_times))
    
    return {
        'cara_tempo': cara_tempo,
        'librosa_tempo': librosa_tempo,
        'tempo_diff': tempo_diff,
        'tempo_rel_diff': tempo_rel_diff,
        'cara_num_beats': len(cara_times),
        'librosa_num_beats': len(librosa_times),
        'beat_alignment_score': beat_alignment_score,
        'mean_beat_error': mean_beat_error,
        'matched_beats': matched_beats,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tolerance': tolerance
    }

def create_comparison_plot(cara_data, librosa_data, comparison, audio_file, output_path="beat_comparison.png"):
    """
    Generate and save a 3x2 figure that visualizes and compares CARA and Librosa beat-tracking results.
    
    Parameters:
        cara_data (dict): CARA results with keys 'beat_times' (array of seconds) and 'beat_frames'.
        librosa_data (dict): Librosa results with keys 'beat_times' (array of seconds) and 'beat_frames'.
        comparison (dict): Comparison metrics used for annotations and panels. Must include
            'cara_tempo', 'librosa_tempo', 'tempo_diff', 'tempo_rel_diff', 'cara_num_beats',
            'librosa_num_beats', 'matched_beats', 'precision', 'recall', 'f1_score',
            'mean_beat_error', 'tolerance', and 'beat_alignment_score'.
        audio_file (str): Path to the audio file used to draw the waveform.
        output_path (str): File path where the PNG figure will be saved.
    
    Behavior:
        Creates six subplots: waveform with beat markers, first-20-beat timing comparison,
        beat-interval histograms, tempo bar chart, beat-alignment matrix (using the provided
        tolerance), and a textual summary panel. The figure is saved to `output_path` and
        a confirmation message is printed. Errors are caught and reported to stdout.
    """
    try:
        # Load audio for visualization
        y, sr = librosa.load(audio_file, sr=None)
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # Plot 1: Waveform with beats
        time_axis = np.linspace(0, len(y) / sr, len(y))
        axes[0, 0].plot(time_axis, y, alpha=0.6, color='gray', linewidth=0.5)
        
        if len(cara_data['beat_times']) > 0:
            axes[0, 0].vlines(cara_data['beat_times'], -1, 1, colors='blue', 
                             linestyle='-', alpha=0.8, linewidth=2, label='CARA')
        
        if len(librosa_data['beat_times']) > 0:
            axes[0, 0].vlines(librosa_data['beat_times'], -1, 1, colors='red', 
                             linestyle='--', alpha=0.8, linewidth=2, label='Librosa')
        
        axes[0, 0].set_title('Audio Waveform with Beat Positions')
        axes[0, 0].set_xlabel('Time (seconds)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Beat timing comparison (first 20 beats)
        max_beats = min(20, len(cara_data['beat_times']), len(librosa_data['beat_times']))
        if max_beats > 0:
            beat_indices = np.arange(max_beats)
            axes[0, 1].plot(beat_indices, cara_data['beat_times'][:max_beats], 
                           'bo-', label='CARA', markersize=6)
            axes[0, 1].plot(beat_indices, librosa_data['beat_times'][:max_beats], 
                           'ro--', label='Librosa', markersize=6)
            axes[0, 1].set_title(f'Beat Timing Comparison (First {max_beats} Beats)')
            axes[0, 1].set_xlabel('Beat Index')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No beats to compare', ha='center', va='center',
                           transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Beat Timing Comparison')
        
        # Plot 3: Beat interval analysis
        if len(cara_data['beat_times']) > 1:
            cara_intervals = np.diff(cara_data['beat_times'])
            axes[1, 0].hist(cara_intervals, bins=20, alpha=0.7, color='blue', 
                           label=f'CARA (mean: {np.mean(cara_intervals):.3f}s)')
        
        if len(librosa_data['beat_times']) > 1:
            librosa_intervals = np.diff(librosa_data['beat_times'])
            axes[1, 0].hist(librosa_intervals, bins=20, alpha=0.7, color='red', 
                           label=f'Librosa (mean: {np.mean(librosa_intervals):.3f}s)')
        
        axes[1, 0].set_title('Beat Interval Distribution')
        axes[1, 0].set_xlabel('Interval (seconds)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Tempo comparison
        tempos = [comparison['cara_tempo'], comparison['librosa_tempo']]
        labels = ['CARA', 'Librosa']
        colors = ['blue', 'red']
        
        bars = axes[1, 1].bar(labels, tempos, color=colors, alpha=0.7)
        axes[1, 1].set_title(f'Tempo Comparison\nDiff: {comparison["tempo_diff"]:.2f} BPM ({comparison["tempo_rel_diff"]:.1f}%)')
        axes[1, 1].set_ylabel('Tempo (BPM)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, tempo in zip(bars, tempos, strict=True):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{tempo:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 5: Beat alignment visualization
        if len(cara_data['beat_times']) > 0 and len(librosa_data['beat_times']) > 0:
            # Create alignment matrix
            tolerance = comparison['tolerance']
            alignment_matrix = np.zeros((len(cara_data['beat_times']), len(librosa_data['beat_times'])))
            
            for i, cara_beat in enumerate(cara_data['beat_times']):
                for j, librosa_beat in enumerate(librosa_data['beat_times']):
                    diff = abs(cara_beat - librosa_beat)
                    if diff <= tolerance:
                        alignment_matrix[i, j] = 1 - (diff / tolerance)  # Closer = higher value
            
            im = axes[2, 0].imshow(alignment_matrix, aspect='auto', cmap='Blues', 
                                  interpolation='nearest')
            axes[2, 0].set_title(f'Beat Alignment Matrix (tolerance: {tolerance*1000:.0f}ms)')
            axes[2, 0].set_xlabel('Librosa Beat Index')
            axes[2, 0].set_ylabel('CARA Beat Index')
            plt.colorbar(im, ax=axes[2, 0])
        else:
            axes[2, 0].text(0.5, 0.5, 'No beats for alignment', ha='center', va='center',
                           transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Beat Alignment Matrix')
        
        # Plot 6: Summary statistics
        axes[2, 1].axis('off')
        
        summary_text = f"""
BEAT TRACKING COMPARISON

📊 Beat Count:
   CARA:     {comparison['cara_num_beats']} beats
   Librosa:  {comparison['librosa_num_beats']} beats

🎵 Tempo Analysis:
   CARA:     {comparison['cara_tempo']:.2f} BPM
   Librosa:  {comparison['librosa_tempo']:.2f} BPM
   Diff:     {comparison['tempo_diff']:.2f} BPM ({comparison['tempo_rel_diff']:.1f}%)

🎯 Beat Alignment:
   Matched:  {comparison['matched_beats']}/{max(comparison['cara_num_beats'], comparison['librosa_num_beats'])}
   Precision: {comparison['precision']:.3f}
   Recall:    {comparison['recall']:.3f}
   F1-Score:  {comparison['f1_score']:.3f}
   Mean Error: {comparison['mean_beat_error']*1000:.1f}ms
   Tolerance:  {comparison['tolerance']*1000:.0f}ms

📈 Overall Score:
   Alignment: {comparison['beat_alignment_score']:.3f}
"""
        
        axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"📊 Beat comparison plot saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error creating comparison plot: {e}")

def print_summary_report(comparison, cara_data, librosa_data, cara_file, audio_file):
    """
    Prints a human-readable summary report of CARA vs Librosa beat-tracking comparison.
    
    Parameters:
        comparison (dict): Metrics produced by analyze_beat_comparison, including keys such as
            'cara_num_beats', 'librosa_num_beats', 'cara_tempo', 'librosa_tempo',
            'tempo_diff', 'tempo_rel_diff', 'tolerance', 'matched_beats',
            'precision', 'recall', 'f1_score', 'mean_beat_error', and 'beat_alignment_score'.
        cara_data (dict): Parsed CARA results dictionary (expects a 'params' sub-dictionary).
        librosa_data (dict): Librosa results dictionary (expects a 'params' sub-dictionary).
        cara_file (str): Path to the CARA output file shown in the report header.
        audio_file (str): Path to the audio file shown in the report header.
    
    Description:
        Outputs a formatted console report that summarizes beat counts, tempos,
        tempo differences, alignment metrics (matched beats, precision, recall, F1,
        mean beat error), parameter comparisons (e.g., tightness and trim), and a
        qualitative assessment of matching quality and tempo agreement.
    """
    print("\n" + "="*80)
    print("🥁 CARA vs LIBROSA BEAT TRACKING COMPARISON REPORT")
    print("="*80)
    
    print(f"📁 Input Files:")
    print(f"   CARA Output: {cara_file}")
    print(f"   Audio File: {audio_file}")
    
    print(f"\n🎯 BEAT TRACKING RESULTS:")
    print(f"   CARA Beats:     {comparison['cara_num_beats']}")
    print(f"   Librosa Beats:  {comparison['librosa_num_beats']}")
    print(f"   CARA Tempo:     {comparison['cara_tempo']:.2f} BPM")
    print(f"   Librosa Tempo:  {comparison['librosa_tempo']:.2f} BPM")
    print(f"   Tempo Diff:     {comparison['tempo_diff']:.2f} BPM ({comparison['tempo_rel_diff']:.1f}%)")
    
    print(f"\n🔍 BEAT ALIGNMENT ANALYSIS:")
    print(f"   Tolerance:      {comparison['tolerance']*1000:.0f} ms")
    print(f"   Matched Beats:  {comparison['matched_beats']}")
    print(f"   Precision:      {comparison['precision']:.3f}")
    print(f"   Recall:         {comparison['recall']:.3f}")
    print(f"   F1-Score:       {comparison['f1_score']:.3f}")
    print(f"   Mean Error:     {comparison['mean_beat_error']*1000:.1f} ms")
    print(f"   Alignment Score: {comparison['beat_alignment_score']:.3f}")
    
    print(f"\n📊 PARAMETER COMPARISON:")
    cara_params = cara_data['params']
    librosa_params = librosa_data['params']
    print(f"   Tightness:    CARA={cara_params.get('tightness', 'N/A')}, Librosa={librosa_params.get('tightness', 'N/A')}")
    print(f"   Trim:         CARA={'Yes' if cara_params.get('trim', False) else 'No'}, Librosa={'Yes' if librosa_params.get('trim', False) else 'No'}")
    
    print(f"\n📈 ASSESSMENT:")
    if comparison['f1_score'] >= 0.9:
        print(f"   🎉 EXCELLENT: F1-Score >= 0.9")
    elif comparison['f1_score'] >= 0.8:
        print(f"   ✅ GOOD: F1-Score >= 0.8")
    elif comparison['f1_score'] >= 0.7:
        print(f"   ⚠️  FAIR: F1-Score >= 0.7")
    else:
        print(f"   ❌ POOR: F1-Score < 0.7")
    
    if comparison['tempo_rel_diff'] < 5.0:
        print(f"   🎵 TEMPO: Excellent agreement (< 5% difference)")
    elif comparison['tempo_rel_diff'] < 10.0:
        print(f"   🎵 TEMPO: Good agreement (< 10% difference)")
    else:
        print(f"   🎵 TEMPO: Poor agreement (>= 10% difference)")
    
    print("="*80)

def main():
    """
    Run the end-to-end CARA vs Librosa beat-tracking comparison pipeline.
    
    Performs argument parsing for input CARA and audio file paths, validates files, loads CARA beat results, computes Librosa beat-tracking with matching parameters, analyzes comparison metrics, generates and saves a multi-panel comparison plot, writes a Librosa beat-reference file, and prints a human-readable summary report.
    
    Returns:
        int: Exit code — `0` on success, non-zero (typically `1`) on failure.
    """
    # Parse command line arguments
    if len(sys.argv) >= 3:
        cara_file = sys.argv[1]
        audio_file = sys.argv[2]
    elif len(sys.argv) == 2:
        cara_file = sys.argv[1]
        audio_file = "files/riad.wav"
    else:
        cara_file = "../outputs/beat_tracking_default.txt"
        audio_file = "files/riad.wav"
    
    print("🥁 CARA vs Librosa Beat Tracking Comparison")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists(cara_file):
        print(f"❌ CARA output file not found: {cara_file}")
        print("💡 Run CARA's test_beat_track first to generate the output file")
        return 1
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return 1
    
    # Load CARA beat results
    cara_data = load_cara_beats(cara_file)
    if cara_data is None:
        return 1
    
    print(f"✅ Loaded CARA beats: {cara_data['results']['num_beats']} beats, {cara_data['results']['tempo_bpm']:.2f} BPM")
    
    # Compute librosa beats with matching parameters
    librosa_data = compute_librosa_beats(audio_file, cara_data['params'])
    if librosa_data is None:
        return 1
    
    # Analyze comparison
    comparison = analyze_beat_comparison(cara_data, librosa_data)
    
    # Create output directory
    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plot
    plot_path = output_dir / "cara_librosa_beat_comparison.png"
    create_comparison_plot(cara_data, librosa_data, comparison, audio_file, str(plot_path))
    
    # Save librosa reference for future use
    librosa_ref_path = output_dir / "librosa_beat_reference.txt"
    with open(librosa_ref_path, 'w') as f:
        f.write(f"# Librosa Beat Tracking Reference\n")
        f.write(f"# Audio file: {audio_file}\n")
        f.write(f"# Parameters: tightness={librosa_data['params']['tightness']}, ")
        f.write(f"trim={librosa_data['params']['trim']}\n")
        f.write(f"# Tempo: {librosa_data['tempo_bpm']:.6f} BPM\n")
        f.write(f"# Beats: {librosa_data['num_beats']}\n")
        f.write(f"# Format: beat_index frame_position time_seconds\n")
        
        for i, (frame, time) in enumerate(zip(librosa_data['beat_frames'], librosa_data['beat_times'], strict=True)):
            f.write(f"{i} {frame} {time:.6f}\n")
    
    print(f"💾 Librosa reference saved to: {librosa_ref_path}")
    
    # Print summary report
    print_summary_report(comparison, cara_data, librosa_data, cara_file, audio_file)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)