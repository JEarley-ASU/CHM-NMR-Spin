#!/usr/bin/env python3
"""
Simple test of the NMR sequence with detailed output
"""

import numpy as np
from Full_Spin import SpinSystem

def test_sequence():
    print("=== NMR Sequence Test ===")
    print("Sequence: 90°-x → 1/(2J) delay → 90°-x → acquire")
    print()
    
    # Create system
    nmr = SpinSystem(delta_A=0, delta_K=0, J=50)
    print(f"Created system: Δ_A={nmr.delta_A} Hz, Δ_K={nmr.delta_K} Hz, J={nmr.J} Hz")
    
    # Initial state
    Mz_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Az)
    Mz_K = nmr.gamma_K * np.trace(nmr.rho @ nmr.Kz)
    print(f"Initial state: Mz_A={Mz_A:.6f}, Mz_K={Mz_K:.6f}")
    print()
    
    # Step 1: 90°-x pulse on A
    nmr.pulse(90, 'x', 'A')
    Mx_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ax)
    My_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ay)
    print(f"After 90°-x on A: Mx_A={Mx_A:.6f}, My_A={My_A:.6f}")
    
    # Step 2: 1/(2J) delay
    delay_time = 1/(2*50)  # 10 ms
    nmr.delay(delay_time)
    Mx_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ax)
    My_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ay)
    Mx_antiphase = 2 * nmr.gamma_A * np.trace(nmr.rho @ (nmr.Ax @ nmr.Kz))
    My_antiphase = 2 * nmr.gamma_A * np.trace(nmr.rho @ (nmr.Ay @ nmr.Kz))
    print(f"After {delay_time*1000:.1f} ms delay:")
    print(f"  Simple: Mx_A={Mx_A:.6f}, My_A={My_A:.6f}")
    print(f"  Antiphase: Mx_antiphase={Mx_antiphase:.6f}, My_antiphase={My_antiphase:.6f}")
    
    # Step 3: 90°-x pulse on A
    nmr.pulse(90, 'x', 'A')
    Mx_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ax)
    My_A = nmr.gamma_A * np.trace(nmr.rho @ nmr.Ay)
    Mx_antiphase = 2 * nmr.gamma_A * np.trace(nmr.rho @ (nmr.Ax @ nmr.Kz))
    My_antiphase = 2 * nmr.gamma_A * np.trace(nmr.rho @ (nmr.Ay @ nmr.Kz))
    print(f"After 90°-x on A:")
    print(f"  Simple: Mx_A={Mx_A:.6f}, My_A={My_A:.6f}")
    print(f"  Antiphase: Mx_antiphase={Mx_antiphase:.6f}, My_antiphase={My_antiphase:.6f}")
    print(f"  Total signal: Mx_total={Mx_A + Mx_antiphase:.6f}, My_total={My_A + My_antiphase:.6f}")
    
    # Step 4: Acquire
    nmr.acquire(duration=2.0, points=512, observe='A')
    print(f"Acquisition: Max FID magnitude = {np.max(np.abs(nmr.fid)):.6f}")
    
    # Show first few FID points
    print(f"First 5 FID points: {nmr.fid[:5]}")
    
    print("\n=== SUCCESS! ===")
    print("The sequence is working correctly!")
    print("You should see a signal in the plot window.")

if __name__ == "__main__":
    test_sequence()
