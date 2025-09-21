import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import io
import pandas as pd

class SpinSystem:
    def __init__(self, delta_A=10.0, delta_K=25.0, J=0.0, T2=0.5,
                 gamma_A=1.0, gamma_K=1.0):  # A=1H (ref), K=13C (~0.251)
        self.delta_A = delta_A
        self.delta_K = delta_K
        self.J = J
        self.T2 = T2
        self.gamma_A = gamma_A*1e-6
        self.gamma_K = gamma_K*1e-6
        self._setup_operators()
        self.reset()
        self.sequence_log = []

    def reset(self):
        """Reset system to thermal equilibrium with γ-weighted polarization"""
        # Thermal Z-magnetization ~ γ * I_z for each spin (up to a constant factor)
        pA = self.gamma_A
        pK = self.gamma_K
        self.rho = (pA/2)*self.Az + (pK/2)*self.Kz
        self.sequence_log = ["Reset to equilibrium (γ-weighted)"]
        
    def _setup_operators(self):
        """Create spin operators for two-spin system"""
        # Pauli matrices
        sx = np.array([[0, 1], [1, 0]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        
        # Two-spin operators (tensor products)
        self.Ax = np.kron(sx, I2) / 2
        self.Ay = np.kron(sy, I2) / 2
        self.Az = np.kron(sz, I2) / 2
        
        self.Kx = np.kron(I2, sx) / 2
        self.Ky = np.kron(I2, sy) / 2
        self.Kz = np.kron(I2, sz) / 2
        #print(self.Kz)
        self.E = np.eye(4, dtype=complex)
        
        # Hamiltonian
        self.H0 = (2*np.pi*self.delta_A*self.Az + 
                   2*np.pi*self.delta_K*self.Kz + 
                   2*np.pi*self.J*self.Az@self.Kz)
        
    def pulse(self, flip_angle, phase=0, spin='AK'):
        """Apply RF pulse"""
        # Convert angle to radians
        angle = np.radians(flip_angle)

        # Convert phase notation
        if phase == 'x':
            phi = 0.0
        elif phase == 'y':
            phi = np.pi/2
        elif phase == '-x':
            phi = np.pi
        elif phase == '-y':
            phi = 3*np.pi/2
        else:
            phi = float(phase)

        # Build rotation operator correctly for spin-1/2:


        if 'A' in spin:
            H_rot_A = np.cos(phi) * self.Ax + np.sin(phi) * self.Ay
            R_A = expm(-1j * angle * H_rot_A)
        else:
            R_A = self.E  # Identity

        # Define rotation operator for spin K
        if 'K' in spin:
            H_rot_K = np.cos(phi) * self.Kx + np.sin(phi) * self.Ky
            R_K = expm(-1j * angle * H_rot_K)
        else:
            R_K = self.E  # Identity

        R = R_A @ R_K
        print(R)

        # Apply rotation
        self.rho = R @ self.rho @ R.conj().T

        # Debug: check magnetization after pulse
        Mx_A = self.gamma_A * np.trace(self.rho @ self.Ax)
        My_A = self.gamma_A * np.trace(self.rho @ self.Ay)
        Mx_K = self.gamma_K * np.trace(self.rho @ self.Kx)
        My_K = self.gamma_K * np.trace(self.rho @ self.Ky)
        print(f"After pulse: Mx_A={Mx_A:.6f}, My_A={My_A:.6f}, Mx_K={Mx_K:.6f}, My_K={My_K:.6f}")

        # Log the pulse
        phase_str = phase if isinstance(phase, str) else f"{np.degrees(phi):.0f}°"
        self.sequence_log.append(f"Pulse: {flip_angle}°_{phase_str} on spin {spin}")
        
    def delay(self, time):
        """
        Free evolution under chemical shifts and J-coupling
        
        Parameters:
        -----------
        time : float
            Delay time in seconds
        """
        if time > 0:
            try:
                # Evolution with relaxation
                U = expm(-1j * self.H0 * time)
                self.rho = U @ self.rho @ U.conj().T
                
                # Apply T2 decay to transverse components
                decay = np.exp(-time / self.T2)
                for i in range(4):
                    for j in range(4):
                        if i != j:
                            self.rho[i, j] *= decay
            except Exception as e:
                print(f"Error in delay calculation: {e}")
                print(f"Delta_A: {self.delta_A}, Delta_K: {self.delta_K}, J: {self.J}")
                        
        self.sequence_log.append(f"Delay: {time*1000:.1f} ms")
        
    def acquire(self, duration=1.0, points=512, observe='AK'):
        """
        Acquire FID
        
        Parameters:
        -----------
        duration : float
            Acquisition time in seconds
        points : int
            Number of points
        observe : str
            Which spin to detect: 'A', 'K', or 'AK'
            
        Returns:
        --------
        t : array
            Time points
        fid : array
            Complex FID signal
        """
        self.time = np.linspace(0, duration, points)
        self.fid = np.zeros(points, dtype=complex)
        
        # Save initial state
        rho0 = self.rho.copy()
        
        for i, ti in enumerate(self.time):
            self.rho = rho0.copy()
            
            if ti > 0:
                try:
                    # Evolve
                    U = expm(-1j * self.H0 * ti)
                    self.rho = U @ self.rho @ U.conj().T
                    
                    # T2 decay
                    decay = np.exp(-ti / self.T2)
                    for m in range(4):
                        for n in range(4):
                            if m != n:
                                self.rho[m, n] *= decay
                except Exception as e:
                    print(f"Error in acquisition evolution: {e}")
                    print(f"Delta_A: {self.delta_A}, Delta_K: {self.delta_K}, J: {self.J}")
            
            # Detect magnetization based on observe parameter
            if observe == 'A':
                Mx = self.gamma_A * np.trace(self.rho @ self.Ax)
                My = self.gamma_A * np.trace(self.rho @ self.Ay)
            elif observe == 'K':
                Mx = self.gamma_K * np.trace(self.rho @ self.Kx)
                My = self.gamma_K * np.trace(self.rho @ self.Ky)
            else:  # 'both'
                Mx = (self.gamma_A * np.trace(self.rho @ self.Ax) +
                      self.gamma_K * np.trace(self.rho @ self.Kx))
                My = (self.gamma_A * np.trace(self.rho @ self.Ay) +
                      self.gamma_K * np.trace(self.rho @ self.Ky))
                
            self.fid[i] = Mx + 1j*My
            
            # Debug: print signal values for first few points
            if i < 3:
                print(f"Point {i}: Mx={Mx:.6f}, My={My:.6f}, |fid|={abs(self.fid[i]):.6f}")
            
        # Check for very small signals and handle appropriately
        max_signal = np.max(np.abs(self.fid))
        print(f"Max signal magnitude: {max_signal:.10f}")
        if max_signal < 1e-10:  # Much more relaxed threshold
            print("Signal too small, zeroing out")
            self.fid = np.zeros_like(self.fid)
        else:
            print(f"Signal detected: {max_signal:.10f}")
            
        self.sequence_log.append(f"Acquire: {duration*1000:.1f} ms, {points} points, observe={observe}")

    def acquire_with_decoupling(self, duration=1.0, points=512, observe='K', decouple='A'):
        """
        Acquire FID with optional decoupling
        
        Parameters:
        -----------
        duration : float
            Acquisition time in seconds
        points : int
            Number of points
        observe : str
            Which spin to detect: 'A', 'K', or 'both'
        decouple : str
            Which spin to decouple: 'A', 'K', or None
            
        Returns:
        --------
        t : array
            Time points
        fid : array
            Complex FID signal
        """
        self.time = np.linspace(0, duration, points)
        self.fid = np.zeros(points, dtype=complex)
        
        # Save initial state
        rho0 = self.rho.copy()
        
        # Build effective Hamiltonian with decoupling
        if decouple == 'A':
            # Remove A spin terms from Hamiltonian (strong irradiation averages to zero)
            H_eff = 2*np.pi*self.delta_K*self.Kz  # Only K chemical shift remains
        elif decouple == 'K':
            H_eff = 2*np.pi*self.delta_A*self.Az  # Only A chemical shift remains
        else:
            H_eff = self.H0  # Full coupled Hamiltonian
        
        for i, ti in enumerate(self.time):
            self.rho = rho0.copy()
            
            if ti > 0:
                # Evolve with effective Hamiltonian
                U = expm(-1j * H_eff * ti)
                self.rho = U @ self.rho @ U.conj().T
                
                # T2 decay
                decay = np.exp(-ti / self.T2)
                for m in range(4):
                    for n in range(4):
                        if m != n:
                            self.rho[m, n] *= decay
            
            # Detect magnetization
            if observe == 'K':
                Mx = self.gamma_K * np.trace(self.rho @ self.Kx)
                My = self.gamma_K * np.trace(self.rho @ self.Ky)
            elif observe == 'A':
                Mx = self.gamma_A * np.trace(self.rho @ self.Ax)
                My = self.gamma_A * np.trace(self.rho @ self.Ay)
            else:
                Mx = (self.gamma_A * np.trace(self.rho @ self.Ax) +
                      self.gamma_K * np.trace(self.rho @ self.Kx))
                My = (self.gamma_A * np.trace(self.rho @ self.Ay) +
                      self.gamma_K * np.trace(self.rho @ self.Ky))
                
            self.fid[i] = Mx + 1j*My
        # Check for very small signals and handle appropriately
        max_signal = np.max(np.abs(self.fid))
        print(f"Max signal magnitude: {max_signal:.10f}")
        if max_signal < 1e-10:  # Much more relaxed threshold
            print("Signal too small, zeroing out")
            self.fid = np.zeros_like(self.fid)
        else:
            print(f"Signal detected: {max_signal:.10f}")
            
        self.sequence_log.append(f"Acquire: {duration*1000:.1f} ms, observe={observe}, decouple={decouple}")

    def plot_1D(self):
        """Plot 1D FID and spectrum"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # FID
        ax1.plot(self.time, np.real(self.fid), 'b-', label='Real')
        ax1.plot(self.time, np.imag(self.fid), 'r--', label='Imaginary')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal')
        ax1.set_title('FID')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits
        if np.max(np.abs(self.fid)) > 0:
            ax1.set_ylim([-np.max(np.abs(self.fid))*1.1, np.max(np.abs(self.fid))*1.1])
        
        # Spectrum
        freq = np.fft.fftfreq(len(self.time), self.time[1]-self.time[0])
        freq = np.fft.fftshift(freq)
        spec = np.fft.fftshift(np.fft.fft(self.fid))
        
        ax2.plot(freq, np.real(spec), 'b-')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Spectrum')
        
        # Dynamic frequency range based on chemical shifts and J-coupling
        # Calculate expected peak positions: delta +/- J/2
        peak_A_high = self.delta_A + abs(self.J)/2
        peak_A_low = self.delta_A - abs(self.J)/2
        peak_K_high = self.delta_K + abs(self.J)/2
        peak_K_low = self.delta_K - abs(self.J)/2
        
        # Find the range that covers all peaks with buffer
        all_peaks = [peak_A_high, peak_A_low, peak_K_high, peak_K_low]
        min_peak = min(all_peaks)
        max_peak = max(all_peaks)
        
        # Add 20% buffer on each side, with minimum range of 20 Hz
        range_size = max(max_peak - min_peak, 20)
        buffer = range_size * 0.2
        ax2.set_xlim([min_peak - buffer, max_peak + buffer])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

test = SpinSystem(delta_A=0, delta_K=0, J=50)

test.pulse(90, 'x', spin="A")
'''
test.delay(1/(2*50))
test.pulse(90, 'y', spin="A")
test.acquire(duration=2.0, points=512, observe="A")
test.plot_1D()
plt.show()
'''