#!/usr/bin/env python3
"""
Test the correct sequence: 90°-x → delay → 90°-x → acquire
"""

import numpy as np
import matplotlib.pyplot as plt
from Full_Spin import SpinSystem

def test_sequence():
    """Test the correct sequence: 90°-x → 1/(2J) delay → 90°-x → acquire"""
    print("Testing 90°-x → delay → 90°-x → acquire sequence")
    
    # Test case: Δ_A = 0, Δ_K = 0, J = 50 Hz
    nmr = SpinSystem(0.0, 0.0, 50.0, 0.5, 1.0, 0.251)
    
    print(f"Initial state: Mz_A={nmr.gamma_A * np.trace(nmr.rho @ nmr.Az):.6f}, Mz_K={nmr.gamma_K * np.trace(nmr.rho @ nmr.Kz):.6f}")
    
    # Step 1: 90°-x pulse on A
    nmr.pulse(90, 'x', 'A')
    Mx_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ax)
    My_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ay)
    print(f"After 90°-x on A: Mx_A={Mx_A:.6f}, My_A={My_A:.6f}")
    
    # Step 2: 1/(2J) delay
    delay_time = 1.0 / (2 * 50.0)  # 10 ms for J=50 Hz
    nmr.delay(delay_time)
    Mx_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ax)
    My_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ay)
    print(f"After {delay_time*1000:.1f} ms delay: Mx_A={Mx_A:.6f}, My_A={My_A:.6f}")
    
    # Check for antiphase terms
    Mx_antiphase = 2 * nmr.gamma_A * np.trace(nmr.rho @ (nmr.Ax @ nmr.Kz))
    My_antiphase = 2 * nmr.gamma_A * np.trace(nmr.rho @ (nmr.Ay @ nmr.Kz))
    print(f"Antiphase terms: Mx_antiphase={Mx_antiphase:.6f}, My_antiphase={My_antiphase:.6f}")
    
    # Step 3: 90°-x pulse on A (not 90°-y!)
    nmr.pulse(90, 'x', 'A')
    Mx_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ax)
    My_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ay)
    Mx_antiphase = 2 * nmr.gamma_A * np.trace(nmr.rho @ (nmr.Ax @ nmr.Kz))
    My_antiphase = 2 * nmr.gamma_A * np.trace(nmr.rho @ (nmr.Ay @ nmr.Kz))
    print(f"After 90°-x on A: Mx_A={Mx_A:.6f}, My_A={My_A:.6f}")
    print(f"Antiphase terms: Mx_antiphase={Mx_antiphase:.6f}, My_antiphase={My_antiphase:.6f}")
    print(f"Total signal: Mx_total={Mx_A + Mx_antiphase:.6f}, My_total={My_A + My_antiphase:.6f}")
    
    # Step 4: Acquire
    nmr.acquire(duration=2.0, points=1200, observe='A')
    
    print(f"FID shape: {nmr.fid.shape}")
    print(f"Max FID magnitude: {np.max(np.abs(nmr.fid)):.6f}")
    print(f"First few FID points: {nmr.fid[:5]}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # FID
    ax1.plot(nmr.time, np.real(nmr.fid), 'b-', label='Real')
    ax1.plot(nmr.time, np.imag(nmr.fid), 'r--', label='Imaginary')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('FID')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Spectrum
    freq = np.fft.fftfreq(len(nmr.time), nmr.time[1]-nmr.time[0])
    freq = np.fft.fftshift(freq)
    spec = np.fft.fftshift(np.fft.fft(nmr.fid))
    
    ax2.plot(freq, np.real(spec), 'b-', label='Real')
    ax2.plot(freq, np.imag(spec), 'r--', label='Imaginary')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Spectrum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-100, 100])
    
    plt.tight_layout()
    plt.savefig('test_sequence_correct_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'test_sequence_correct_results.png'")
    
    return nmr

if __name__ == "__main__":
    nmr = test_sequence()
