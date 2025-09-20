import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

class SpinSystem:
    def __init__(self, delta_A=10.0, delta_K=25.0, J=0.0, T2=0.5,
                 gamma_A=1.0, gamma_K=1/3.98):  # A=1H (ref), K=13C (~0.251)
        self.delta_A = delta_A
        self.delta_K = delta_K
        self.J = J
        self.T2 = T2
        self.gamma_A = gamma_A
        self.gamma_K = gamma_K
        self._setup_operators()
        self.reset()
        self.sequence_log = []

    def reset(self):
        """Reset system to thermal equilibrium with γ-weighted polarization"""
        # Thermal Z-magnetization ~ γ * I_z for each spin (up to a constant factor)
        pA = self.gamma_A
        pK = self.gamma_K
        self.rho = (self.E + pA*self.Az + pK*self.Kz) / 4
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
        # R = cos(theta/2)*I - 2i*sin(theta/2)*(cosφ * Sx + sinφ * Sy)
        cos_half = np.cos(angle/2)
        sin_half = np.sin(angle/2)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        if 'A' in spin:
            R_A = cos_half * self.E - 2j * sin_half * (cos_phi * self.Ax + sin_phi * self.Ay)
        else:
            R_A = self.E

        if 'K' in spin:
            R_K = cos_half * self.E - 2j * sin_half * (cos_phi * self.Kx + sin_phi * self.Ky)
        else:
            R_K = self.E

        R = R_A @ R_K

        # Apply rotation
        self.rho = R @ self.rho @ R.conj().T

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
            # Evolution with relaxation
            U = expm(-1j * self.H0 * time)
            self.rho = U @ self.rho @ U.conj().T
            
            # Apply T2 decay to transverse components
            decay = np.exp(-time / self.T2)
            for i in range(4):
                for j in range(4):
                    if i != j:
                        self.rho[i, j] *= decay
                        
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
                # Evolve
                U = expm(-1j * self.H0 * ti)
                self.rho = U @ self.rho @ U.conj().T
                
                # T2 decay
                decay = np.exp(-ti / self.T2)
                for m in range(4):
                    for n in range(4):
                        if m != n:
                            self.rho[m, n] *= decay
            
            # Detect magnetization based on observe parameter
            if observe == 'A':
                # Simple transverse magnetization
                Mx = self.gamma_A * np.trace(self.rho @ self.Ax)
                My = self.gamma_A * np.trace(self.rho @ self.Ay)
                
                # Antiphase magnetization (observable in J-coupled systems)
                # 2I_Ax I_Kz and 2I_Ay I_Kz terms
                Mx_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ax @ self.Kz))
                My_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ay @ self.Kz))
                
                # Total signal (in-phase + antiphase)
                Mx_total = Mx + Mx_antiphase
                My_total = My + My_antiphase
                
            elif observe == 'K':
                # Simple transverse magnetization
                Mx = self.gamma_K * np.trace(self.rho @ self.Kx)
                My = self.gamma_K * np.trace(self.rho @ self.Ky)
                
                # Antiphase magnetization
                Mx_antiphase = 2 * self.gamma_K * np.trace(self.rho @ (self.Kx @ self.Az))
                My_antiphase = 2 * self.gamma_K * np.trace(self.rho @ (self.Ky @ self.Az))
                
                Mx_total = Mx + Mx_antiphase
                My_total = My + My_antiphase
                
            else:  # 'both'
                # Sum of both spins
                Mx_A = self.gamma_A * np.trace(self.rho @ self.Ax)
                My_A = self.gamma_A * np.trace(self.rho @ self.Ay)
                Mx_K = self.gamma_K * np.trace(self.rho @ self.Kx)
                My_K = self.gamma_K * np.trace(self.rho @ self.Ky)
                
                # Antiphase terms
                Mx_A_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ax @ self.Kz))
                My_A_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ay @ self.Kz))
                Mx_K_antiphase = 2 * self.gamma_K * np.trace(self.rho @ (self.Kx @ self.Az))
                My_K_antiphase = 2 * self.gamma_K * np.trace(self.rho @ (self.Ky @ self.Az))
                
                Mx_total = Mx_A + Mx_K + Mx_A_antiphase + Mx_K_antiphase
                My_total = My_A + My_K + My_A_antiphase + My_K_antiphase
                
            self.fid[i] = My_total + 1j*Mx_total
            
        if np.max(self.fid) < 1e-6:
            self.fid = np.zeros_like(self.fid)
            
        self.sequence_log.append(f"Acquire: {duration*1000:.1f} ms, {points} points, observe={observe}")

    
    def plot_1D(self):
        """Plot 1D FID and spectrum"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # FID
        ax1.plot(self.time, np.real(self.fid), 'b-', label='Real')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Signal')
        ax1.set_title('FID')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Spectrum
        freq = np.fft.fftfreq(len(self.time), self.time[1]-self.time[0])
        freq = np.fft.fftshift(freq)
        spec = np.fft.fftshift(np.fft.fft(self.fid))
        
        ax2.plot(freq, np.real(spec), 'b-')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Intensity')
        ax2.set_title('Spectrum')
        ax2.set_xlim([-50, 50])
        ax2.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.show()
        
        
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
            
            # Detect magnetization with antiphase terms
            if observe == 'K':
                # Simple transverse magnetization
                Mx = self.gamma_K * np.trace(self.rho @ self.Kx)
                My = self.gamma_K * np.trace(self.rho @ self.Ky)
                
                # Antiphase magnetization
                Mx_antiphase = 2 * self.gamma_K * np.trace(self.rho @ (self.Kx @ self.Az))
                My_antiphase = 2 * self.gamma_K * np.trace(self.rho @ (self.Ky @ self.Az))
                
                Mx_total = Mx + Mx_antiphase
                My_total = My + My_antiphase
                
            elif observe == 'A':
                # Simple transverse magnetization
                Mx = self.gamma_A * np.trace(self.rho @ self.Ax)
                My = self.gamma_A * np.trace(self.rho @ self.Ay)
                
                # Antiphase magnetization
                Mx_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ax @ self.Kz))
                My_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ay @ self.Kz))
                
                Mx_total = Mx + Mx_antiphase
                My_total = My + My_antiphase
                
            else:  # 'both'
                # Sum of both spins
                Mx_A = self.gamma_A * np.trace(self.rho @ self.Ax)
                My_A = self.gamma_A * np.trace(self.rho @ self.Ay)
                Mx_K = self.gamma_K * np.trace(self.rho @ self.Kx)
                My_K = self.gamma_K * np.trace(self.rho @ self.Ky)
                
                # Antiphase terms
                Mx_A_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ax @ self.Kz))
                My_A_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ay @ self.Kz))
                Mx_K_antiphase = 2 * self.gamma_K * np.trace(self.rho @ (self.Kx @ self.Az))
                My_K_antiphase = 2 * self.gamma_K * np.trace(self.rho @ (self.Ky @ self.Az))
                
                Mx_total = Mx_A + Mx_K + Mx_A_antiphase + Mx_K_antiphase
                My_total = My_A + My_K + My_A_antiphase + My_K_antiphase
                
            self.fid[i] = My_total + 1j*Mx_total
        if np.max(self.fid) < 1e-6:
            self.fid = np.zeros_like(self.fid)
            
        self.sequence_log.append(f"Acquire: {duration*1000:.1f} ms, observe={observe}, decouple={decouple}")



# Test the correct sequence: 90°-x → 1/(2J) delay → 90°-x → acquire
nmr = SpinSystem(delta_A=0, delta_K=0, J=50)

# Correct delay time for 1/(2J)
tt = 1/(2*50)  # 10 ms for J=50 Hz
print(f"Delay time: {tt*1000:.1f} ms")

nmr.reset()
print("Initial state reset")

# Step 1: 90°-x pulse on A
nmr.pulse(90, 'x', spin="A")
print("Applied 90°-x pulse on A")

# Step 2: 1/(2J) delay
nmr.delay(tt)
print(f"Applied {tt*1000:.1f} ms delay")

# Step 3: 90°-x pulse on A (not 90°-y!)
nmr.pulse(90, 'x', spin="A")
print("Applied 90°-x pulse on A")

# Step 4: Acquire
nmr.acquire(duration=2.0, points=512, observe="A")
print("Acquisition completed")

# Plot results
nmr.plot_1D()


    