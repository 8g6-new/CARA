#!/usr/bin/env python3
"""
CARA vs Librosa Tempo Estimation Comparison Tool

This script compares CARA's tempo estimation output with librosa's
implementation, providing comprehensive analysis and validation.

Usage:
    python compare_cara_librosa_tempo.py [cara_output.txt] [audio_file.wav]

If no arguments provided, uses default paths:
    - CARA output: ../outputs/tempo_estimation_default.txt
    - Audio file: files/riad.wav
"""

import sys
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path

def load_cara_tempo(cara_file):
    """
    Load CARA's tempo estimation from text file.
    
    Args:
        cara_file (str): Path to CARA's tempo output file
        
    Returns:
        dict: CARA tempo results with metadata
    """
    try:
        # Parse the header to extract parameters and results
        params = {}
        results = {}
        tempo_values = []
        
        with open(cara_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    # Parse parameter and result lines
                    if 'start_bpm:' in line:
                        params['start_bpm'] = float(line.split(':')[1].strip())
                    elif 'std_bpm:' in line:
                        params['std_bpm'] = float(line.split(':')[1].strip())
                    elif 'max_tempo:' in line:
                        params['max_tempo'] = float(line.split(':')[1].strip())
                    elif 'ac_size:' in line:
                        params['ac_size'] = float(line.split(':')[1].split()[0])
                    elif 'use_prior:' in line:
                        params['use_prior'] = 'true' in line.lower()
                    elif 'aggregate:' in line:
                        params['aggregate'] = 'true' in line.lower()
                    elif 'length:' in line:
                        results['length'] = int(line.split(':')[1].split()[0])
                    elif 'frame_rate:' in line:
                        results['frame_rate'] = float(line.split(':')[1].split()[0])
                    elif 'confidence:' in line:
                        results['confidence'] = float(line.split(':')[1].strip())
                elif line and not line.startswith('#'):
                    # Parse tempo data
                    parts = line.split()
                    if len(parts) == 2:
                        tempo_values.append(float(parts[1]))
        
        return {
            'params': params,
            'results': results,
            'tempo_bpm': np.array(tempo_values),
            'filename': cara_file
        }
        
    except Exception as e:
        print(f"❌ Error loading CARA tempo from {cara_file}: {e}")
        return None

def compute_librosa_tempo(audio_file, cara_params=None):
    """
    Compute librosa tempo estimation with parameters matching CARA.
    
    Args:
        audio_file (str): Path to audio file
        cara_params (dict): CARA parameters to match
        
    Returns:
        dict: Librosa tempo results
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=None)
        print(f"📁 Loaded audio: {len(y)} samples, {sr} Hz, {len(y)/sr:.3f}s")
        
        # STFT parameters (matching CARA's likely configuration)
        n_fft = 2048
        hop_length = 512
        n_mels = 128
        
        print(f"🔧 STFT parameters: n_fft={n_fft}, hop_length={hop_length}")
        
        # Compute onset strength envelope
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr, 
            n_fft=n_fft, 
            hop_length=hop_length,
            n_mels=n_mels,
            center=False  # Match CARA's frame alignment
        )
        
        print(f"📊 Onset envelope: {len(onset_env)} frames")
        
        # Set tempo estimation parameters
        if cara_params:
            start_bpm = cara_params.get('start_bpm', 120.0)
            std_bpm = cara_params.get('std_bpm', 1.0)
            max_tempo = cara_params.get('max_tempo', 320.0)
            ac_size = cara_params.get('ac_size', 8.0)
            use_prior = cara_params.get('use_prior', True)
        else:
            start_bpm = 120.0
            std_bpm = 1.0
            max_tempo = 320.0
            ac_size = 8.0
            use_prior = True
        
        # Compute tempo with matching parameters
        if use_prior:
            # Use default log-normal prior
            tempo_bpm = librosa.feature.tempo(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop_length,
                start_bpm=start_bpm,
                std_bpm=std_bpm,
                ac_size=ac_size,
                max_tempo=max_tempo
            )
        else:
            # Use uniform prior (no prior weighting)
            import scipy.stats
            prior = scipy.stats.uniform(30, max_tempo - 30)
            tempo_bpm = librosa.feature.tempo(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=hop_length,
                ac_size=ac_size,
                max_tempo=max_tempo,
                prior=prior
            )
        
        # Also compute autocorrelation for analysis
        win_length = int(ac_size * sr / hop_length)
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=hop_length,
            win_length=win_length
        )
        
        # Get autocorrelation (mean over time)
        autocorr = np.mean(tempogram, axis=1)
        
        # Get BPM frequencies
        bpm_freqs = librosa.tempo_frequencies(
            len(autocorr), 
            sr=sr, 
            hop_length=hop_length
        )
        
        print(f"🎯 Librosa tempo: {tempo_bpm[0]:.2f} BPM")
        
        return {
            'tempo_bpm': tempo_bpm,
            'onset_env': onset_env,
            'autocorr': autocorr,
            'bpm_freqs': bpm_freqs,
            'tempogram': tempogram,
            'params': {
                'start_bpm': start_bpm,
                'std_bpm': std_bpm,
                'max_tempo': max_tempo,
                'ac_size': ac_size,
                'use_prior': use_prior,
                'sr': sr,
                'hop_length': hop_length
            }
        }
        
    except Exception as e:
        print(f"❌ Error computing librosa tempo from {audio_file}: {e}")
        return None

def analyze_tempo_comparison(cara_data, librosa_data):
    """
    Analyze tempo estimation comparison between CARA and librosa.
    
    Args:
        cara_data (dict): CARA tempo results
        librosa_data (dict): Librosa tempo results
        
    Returns:
        dict: Comparison metrics
    """
    cara_tempo = cara_data['tempo_bpm'][0] if len(cara_data['tempo_bpm']) > 0 else 0
    librosa_tempo = librosa_data['tempo_bpm'][0]
    
    # Calculate metrics
    absolute_diff = abs(cara_tempo - librosa_tempo)
    relative_diff = absolute_diff / librosa_tempo * 100 if librosa_tempo > 0 else float('inf')
    
    # Check for octave errors (common in tempo estimation)
    octave_ratios = [0.5, 2.0, 1.5, 2/3, 3.0, 1/3]  # Common tempo multiples
    octave_errors = []
    
    for ratio in octave_ratios:
        predicted_tempo = librosa_tempo * ratio
        diff = abs(cara_tempo - predicted_tempo)
        rel_diff = diff / predicted_tempo * 100 if predicted_tempo > 0 else float('inf')
        octave_errors.append({
            'ratio': ratio,
            'predicted_tempo': predicted_tempo,
            'absolute_diff': diff,
            'relative_diff': rel_diff
        })
    
    # Find best octave match
    best_octave = min(octave_errors, key=lambda x: x['absolute_diff'])
    
    return {
        'cara_tempo': cara_tempo,
        'librosa_tempo': librosa_tempo,
        'absolute_diff': absolute_diff,
        'relative_diff': relative_diff,
        'octave_errors': octave_errors,
        'best_octave_match': best_octave,
        'is_octave_error': best_octave['absolute_diff'] < absolute_diff,
        'cara_confidence': cara_data['results'].get('confidence', 0.0)
    }

def create_comparison_plot(cara_data, librosa_data, comparison, output_path="tempo_comparison.png"):
    """
    Create comprehensive tempo comparison plot.
    
    Args:
        cara_data (dict): CARA tempo results
        librosa_data (dict): Librosa tempo results
        comparison (dict): Comparison metrics
        output_path (str): Output plot path
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Tempo comparison bar chart
    tempos = [comparison['cara_tempo'], comparison['librosa_tempo']]
    labels = ['CARA', 'Librosa']
    colors = ['blue', 'red']
    
    bars = axes[0, 0].bar(labels, tempos, color=colors, alpha=0.7)
    axes[0, 0].set_title(f'Tempo Comparison\nDifference: {comparison["absolute_diff"]:.2f} BPM ({comparison["relative_diff"]:.1f}%)')
    axes[0, 0].set_ylabel('Tempo (BPM)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, tempo in zip(bars, tempos):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{tempo:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Autocorrelation comparison (if available)
    if 'autocorr' in librosa_data and len(librosa_data['bpm_freqs']) > 0:
        # Limit to reasonable BPM range for visualization
        max_bpm_plot = 400
        valid_idx = librosa_data['bpm_freqs'] <= max_bpm_plot
        
        axes[0, 1].plot(librosa_data['bpm_freqs'][valid_idx], 
                       librosa_data['autocorr'][valid_idx], 
                       'r-', label='Librosa Autocorr', alpha=0.8)
        axes[0, 1].axvline(comparison['librosa_tempo'], color='red', linestyle='--', 
                          alpha=0.8, label=f'Librosa: {comparison["librosa_tempo"]:.1f} BPM')
        axes[0, 1].axvline(comparison['cara_tempo'], color='blue', linestyle='--', 
                          alpha=0.8, label=f'CARA: {comparison["cara_tempo"]:.1f} BPM')
        axes[0, 1].set_title('Autocorrelation vs BPM')
        axes[0, 1].set_xlabel('BPM')
        axes[0, 1].set_ylabel('Autocorrelation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'Autocorrelation data\nnot available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Autocorrelation Analysis')
    
    # Plot 3: Octave error analysis
    octave_ratios = [err['ratio'] for err in comparison['octave_errors']]
    octave_diffs = [err['absolute_diff'] for err in comparison['octave_errors']]
    
    bars = axes[1, 0].bar(range(len(octave_ratios)), octave_diffs, alpha=0.7)
    axes[1, 0].set_title('Octave Error Analysis')
    axes[1, 0].set_xlabel('Tempo Ratio')
    axes[1, 0].set_ylabel('Absolute Difference (BPM)')
    axes[1, 0].set_xticks(range(len(octave_ratios)))
    axes[1, 0].set_xticklabels([f'{r:.2f}' for r in octave_ratios], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Highlight best match
    best_idx = octave_diffs.index(min(octave_diffs))
    bars[best_idx].set_color('green')
    bars[best_idx].set_alpha(0.9)
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    
    summary_text = f"""
TEMPO ESTIMATION COMPARISON

📊 Results:
   CARA Tempo:     {comparison['cara_tempo']:.2f} BPM
   Librosa Tempo:  {comparison['librosa_tempo']:.2f} BPM
   
📈 Accuracy:
   Absolute Diff:  {comparison['absolute_diff']:.2f} BPM
   Relative Diff:  {comparison['relative_diff']:.1f}%
   CARA Confidence: {comparison['cara_confidence']:.4f}

🎯 Octave Analysis:
   Best Match Ratio: {comparison['best_octave_match']['ratio']:.2f}
   Best Match Tempo: {comparison['best_octave_match']['predicted_tempo']:.1f} BPM
   Best Match Diff:  {comparison['best_octave_match']['absolute_diff']:.2f} BPM
   Is Octave Error:  {'Yes' if comparison['is_octave_error'] else 'No'}

🔧 Parameters:
   Start BPM:      {cara_data['params'].get('start_bpm', 'N/A')}
   Use Prior:      {'Yes' if cara_data['params'].get('use_prior', False) else 'No'}
   Max Tempo:      {cara_data['params'].get('max_tempo', 'N/A')} BPM
   AC Size:        {cara_data['params'].get('ac_size', 'N/A')} sec
"""
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Tempo comparison plot saved to: {output_path}")

def print_summary_report(comparison, cara_data, librosa_data, cara_file, audio_file):
    """
    Print comprehensive summary report.
    
    Args:
        comparison (dict): Comparison metrics
        cara_data (dict): CARA results
        librosa_data (dict): Librosa results
        cara_file (str): CARA output file path
        audio_file (str): Audio file path
    """
    print("\n" + "="*80)
    print("🎵 CARA vs LIBROSA TEMPO ESTIMATION COMPARISON REPORT")
    print("="*80)
    
    print(f"📁 Input Files:")
    print(f"   CARA Output: {cara_file}")
    print(f"   Audio File: {audio_file}")
    
    print(f"\n🎯 TEMPO ESTIMATION RESULTS:")
    print(f"   CARA Tempo:     {comparison['cara_tempo']:.2f} BPM")
    print(f"   Librosa Tempo:  {comparison['librosa_tempo']:.2f} BPM")
    print(f"   Absolute Diff:  {comparison['absolute_diff']:.2f} BPM")
    print(f"   Relative Diff:  {comparison['relative_diff']:.1f}%")
    print(f"   CARA Confidence: {comparison['cara_confidence']:.4f}")
    
    print(f"\n🔍 OCTAVE ERROR ANALYSIS:")
    if comparison['is_octave_error']:
        best = comparison['best_octave_match']
        print(f"   ⚠️  Potential octave error detected!")
        print(f"   Best match ratio: {best['ratio']:.2f}x")
        print(f"   Best match tempo: {best['predicted_tempo']:.1f} BPM")
        print(f"   Best match diff:  {best['absolute_diff']:.2f} BPM ({best['relative_diff']:.1f}%)")
    else:
        print(f"   ✅ No significant octave error detected")
        print(f"   Direct comparison is most accurate")
    
    print(f"\n📊 PARAMETER COMPARISON:")
    cara_params = cara_data['params']
    librosa_params = librosa_data['params']
    print(f"   Start BPM:    CARA={cara_params.get('start_bpm', 'N/A')}, Librosa={librosa_params.get('start_bpm', 'N/A')}")
    print(f"   Use Prior:    CARA={'Yes' if cara_params.get('use_prior', False) else 'No'}, Librosa={'Yes' if librosa_params.get('use_prior', False) else 'No'}")
    print(f"   Max Tempo:    CARA={cara_params.get('max_tempo', 'N/A')}, Librosa={librosa_params.get('max_tempo', 'N/A')}")
    print(f"   AC Size:      CARA={cara_params.get('ac_size', 'N/A')}s, Librosa={librosa_params.get('ac_size', 'N/A')}s")
    
    print(f"\n📈 ASSESSMENT:")
    if comparison['relative_diff'] < 5.0:
        print(f"   🎉 EXCELLENT: Relative difference < 5%")
    elif comparison['relative_diff'] < 10.0:
        print(f"   ✅ GOOD: Relative difference < 10%")
    elif comparison['relative_diff'] < 20.0:
        print(f"   ⚠️  FAIR: Relative difference < 20%")
    else:
        if comparison['is_octave_error']:
            print(f"   🔄 OCTAVE ERROR: Check for tempo doubling/halving")
        else:
            print(f"   ❌ POOR: Large difference detected")
    
    print("="*80)

def main():
    """
    Main function to run the tempo comparison.
    """
    # Parse command line arguments
    if len(sys.argv) >= 3:
        cara_file = sys.argv[1]
        audio_file = sys.argv[2]
    elif len(sys.argv) == 2:
        cara_file = sys.argv[1]
        audio_file = "files/riad.wav"
    else:
        cara_file = "../outputs/tempo_estimation_default.txt"
        audio_file = "files/riad.wav"
    
    print("🎵 CARA vs Librosa Tempo Estimation Comparison")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists(cara_file):
        print(f"❌ CARA output file not found: {cara_file}")
        print("💡 Run CARA's test_tempo first to generate the output file")
        return 1
    
    if not os.path.exists(audio_file):
        print(f"❌ Audio file not found: {audio_file}")
        return 1
    
    # Load CARA tempo results
    cara_data = load_cara_tempo(cara_file)
    if cara_data is None:
        return 1
    
    print(f"✅ Loaded CARA tempo: {cara_data['tempo_bpm'][0]:.2f} BPM")
    
    # Compute librosa tempo with matching parameters
    librosa_data = compute_librosa_tempo(audio_file, cara_data['params'])
    if librosa_data is None:
        return 1
    
    # Analyze comparison
    comparison = analyze_tempo_comparison(cara_data, librosa_data)
    
    # Create output directory
    output_dir = Path("../outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Create comparison plot
    plot_path = output_dir / "cara_librosa_tempo_comparison.png"
    create_comparison_plot(cara_data, librosa_data, comparison, str(plot_path))
    
    # Save librosa reference for future use
    librosa_ref_path = output_dir / "librosa_tempo_reference.txt"
    with open(librosa_ref_path, 'w') as f:
        f.write(f"# Librosa Tempo Estimation Reference\n")
        f.write(f"# Audio file: {audio_file}\n")
        f.write(f"# Parameters: start_bpm={librosa_data['params']['start_bpm']}, ")
        f.write(f"max_tempo={librosa_data['params']['max_tempo']}, ")
        f.write(f"ac_size={librosa_data['params']['ac_size']}\n")
        f.write(f"# Tempo: {librosa_data['tempo_bpm'][0]:.6f} BPM\n")
        f.write(f"0 {librosa_data['tempo_bpm'][0]:.6f}\n")
    
    print(f"💾 Librosa reference saved to: {librosa_ref_path}")
    
    # Print summary report
    print_summary_report(comparison, cara_data, librosa_data, cara_file, audio_file)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
