import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import io

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
        
        # Debug: check initial state
        Mz_A = self.gamma_A * np.trace(self.rho @ self.Az)
        Mz_K = self.gamma_K * np.trace(self.rho @ self.Kz)
        print(f"Initial state: Mz_A={Mz_A:.6f}, Mz_K={Mz_K:.6f}")
        print(f"Parameters: delta_A={self.delta_A}, delta_K={self.delta_K}, J={self.J}")
        print(f"Gamma values: gamma_A={self.gamma_A}, gamma_K={self.gamma_K}")
        
        # Check if density matrix is properly normalized
        trace_rho = np.trace(self.rho)
        print(f"Density matrix trace: {trace_rho:.6f}")
        
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
        self.H0 = (2*np.pi*self.delta_A*self.Az + 2*np.pi*self.delta_K*self.Kz + 2*np.pi*self.J*(self.Ax@self.Kx + self.Ay@self.Ky + self.Az@self.Kz))

        
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
                # Simple transverse magnetization
                Mx = self.gamma_A * np.trace(self.rho @ self.Ax)
                My = self.gamma_A * np.trace(self.rho @ self.Ay)
                
                # Antiphase magnetization (observable in J-coupled systems)
                # 2I_Ax I_Kz and 2I_Ay I_Kz terms
                Mx_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ax @ self.Kz))
                My_antiphase = 2 * self.gamma_A * np.trace(self.rho @ (self.Ay @ self.Kz))
                
                # Additional coherence terms that might be created
                # 2I_Az I_Kx and 2I_Az I_Ky terms
                Mx_coherence = 2 * self.gamma_A * np.trace(self.rho @ (self.Az @ self.Kx))
                My_coherence = 2 * self.gamma_A * np.trace(self.rho @ (self.Az @ self.Ky))
                
                # Detect the specific coherences present in the density matrix
                # These are single-quantum coherences for spin A: |00⟩↔|10⟩ and |01⟩↔|11⟩
                # We need to detect these directly from the density matrix elements
                coherence_01 = self.rho[0,2] + self.rho[2,0]  # |00⟩↔|10⟩
                coherence_23 = self.rho[1,3] + self.rho[3,1]  # |01⟩↔|11⟩
                
                # These coherences contribute to transverse magnetization
                Mx_coherence_direct = self.gamma_A * (coherence_01 + coherence_23).real
                My_coherence_direct = self.gamma_A * (coherence_01 + coherence_23).imag
                
                # Total signal (in-phase + antiphase + additional coherences + direct coherences)
                Mx_total = Mx + Mx_antiphase + Mx_coherence + Mx_coherence_direct
                My_total = My + My_antiphase + My_coherence + My_coherence_direct
                
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

def initialize_session_state():
    """Initialize session state variables"""
    if 'nmr' not in st.session_state:
        st.session_state.nmr = SpinSystem()
    if 'sequence_log' not in st.session_state:
        st.session_state.sequence_log = []
    if 'current_sequence' not in st.session_state:
        st.session_state.current_sequence = []

def replay_sequence():
    """Replay the current pulse sequence"""
    if st.session_state.current_sequence:
        st.session_state.nmr.reset()
        for operation in st.session_state.current_sequence:
            if operation['type'] == 'pulse':
                st.session_state.nmr.pulse(operation['flip_angle'], operation['phase'], operation['spin'])
            elif operation['type'] == 'delay':
                st.session_state.nmr.delay(operation['time'])
            elif operation['type'] == 'acquire':
                if operation.get('decouple') is None:
                    st.session_state.nmr.acquire(operation['duration'], operation['points'], operation['observe'])
                else:
                    st.session_state.nmr.acquire_with_decoupling(operation['duration'], operation['points'], operation['observe'], operation['decouple'])
        # Always update the sequence log after replaying
        st.session_state.sequence_log = st.session_state.nmr.sequence_log.copy()

def main():
    st.set_page_config(
        page_title="NMR Spin System Simulator",
        page_icon="🧲",
        layout="wide"
    )
    
    st.title("🧲 NMR Spin System Simulator")
    st.markdown("Interactive simulation of two-spin NMR system with pulse sequences")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar for parameters
    with st.sidebar:
        st.header("NMR Parameters")
        
        # Chemical shifts
        delta_A = st.number_input("Δ_A (Hz)", value=0.0, step=0.1, format="%.1f", help="Chemical shift of spin A")
        delta_K = st.number_input("Δ_K (Hz)", value=0.0, step=0.1, format="%.1f", help="Chemical shift of spin K")
        
        # J-coupling
        J = st.number_input("J-coupling (Hz)", value=0.0, step=0.1, format="%.1f", help="Scalar coupling constant")
        
        # Round values to 1 decimal place
        delta_A = round(delta_A, 1)
        delta_K = round(delta_K, 1)
        J = round(J, 1)
        
        # Relaxation (hidden, using default)
        T2 = 0.5  # Default value, not shown in UI
        
        # Gyromagnetic ratios
        gamma_A = st.slider("γ_A", 0.25, 4.0, 1.0, 0.01, help="Gyromagnetic ratio of spin A")
        gamma_K = st.slider("γ_K", 0.25, 4.0, 0.251, 0.001, help="Gyromagnetic ratio of spin K")
        
        # Update system parameters
        if st.button("Update Parameters", type="primary"):
            st.session_state.nmr = SpinSystem(
                delta_A=delta_A, delta_K=delta_K, J=J, T2=T2,
                gamma_A=gamma_A, gamma_K=gamma_K
            )
            # Replay current sequence if one exists
            if st.session_state.current_sequence:
                replay_sequence()
                st.success("Parameters updated and sequence replayed!")
            else:
                st.session_state.sequence_log = st.session_state.nmr.sequence_log.copy()
                st.success("Parameters updated!")
        
        # Reset button
        if st.button("Reset System", type="secondary"):
            st.session_state.nmr.reset()
            st.session_state.sequence_log = st.session_state.nmr.sequence_log.copy()
            st.session_state.current_sequence = []  # Clear current sequence
            st.success("System reset to equilibrium!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Pulse Sequence Controls")
        
        # Pulse controls
        st.subheader("RF Pulses")
        pulse_col1, pulse_col2 = st.columns(2)
        
        with pulse_col1:
            flip_angle = st.selectbox("Flip Angle", [90, 180], index=0)
            phase = st.selectbox("Phase", ['x', 'y', '-x', '-y'], index=1)
        
        with pulse_col2:
            spin = st.selectbox("Spin", ['A', 'K', 'AK'], index=2)
            if st.button("Apply Pulse", type="secondary"):
                st.session_state.nmr.pulse(flip_angle, phase, spin)
                # Store operation in current sequence
                st.session_state.current_sequence.append({
                    'type': 'pulse',
                    'flip_angle': flip_angle,
                    'phase': phase,
                    'spin': spin
                })
                st.session_state.sequence_log = st.session_state.nmr.sequence_log.copy()
                st.success(f"Applied {flip_angle}°_{phase} pulse on {spin}")
        
        # Delay controls (moved under pulse panel)
        st.subheader("Delays")
        delay_col1, delay_col2 = st.columns(2)
        
        with delay_col1:
            delay_time = st.number_input("Delay (ms)", 0.0, 1000.0, 0.0, 1.0) / 1000.0
        
        with delay_col2:
            if st.button("Apply Delay", type="secondary"):
                st.session_state.nmr.delay(delay_time)
                # Store operation in current sequence
                st.session_state.current_sequence.append({
                    'type': 'delay',
                    'time': delay_time
                })
                st.session_state.sequence_log = st.session_state.nmr.sequence_log.copy()
                st.success(f"Applied {delay_time*1000:.1f} ms delay")
        
        # Acquisition controls
        st.subheader("Acquisition")
        acq_col1, acq_col2 = st.columns(2)
        
        with acq_col1:
            # Hidden defaults
            duration = 2.0  # Default 2 seconds
            points = 1200   # Default 1200 points
            observe = st.selectbox("Observe", ['A', 'K', 'AK'], index=1)
        
        with acq_col2:
            decouple = st.selectbox("Decouple", ['None', 'A', 'K'], index=0)
            if decouple == 'None':
                decouple = None
            
            if st.button("Acquire", type="primary"):
                if decouple is None:
                    st.session_state.nmr.acquire(duration, points, observe)
                else:
                    st.session_state.nmr.acquire_with_decoupling(duration, points, observe, decouple)
                # Store operation in current sequence
                st.session_state.current_sequence.append({
                    'type': 'acquire',
                    'duration': duration,
                    'points': points,
                    'observe': observe,
                    'decouple': decouple
                })
                st.session_state.sequence_log = st.session_state.nmr.sequence_log.copy()
                st.success("Acquisition completed!")
    
    with col2:
        st.header("Sequence Log")
        for i, step in enumerate(st.session_state.sequence_log):
            st.text(f"{i+1}. {step}")
        
        # Undo and Clear buttons
        undo_col, clear_col = st.columns(2)
        
        with undo_col:
            if st.button("Undo Last", type="secondary"):
                if st.session_state.current_sequence:
                    # Remove last operation from current sequence
                    st.session_state.current_sequence.pop()
                    # Replay the remaining sequence
                    if st.session_state.current_sequence:
                        replay_sequence()
                    else:
                        # If no operations left, just reset
                        st.session_state.nmr.reset()
                        st.session_state.sequence_log = st.session_state.nmr.sequence_log.copy()
                    st.success("Last operation undone!")
                    st.rerun()
                else:
                    st.warning("No operations to undo!")
        
        with clear_col:
            if st.button("Clear All", type="secondary"):
                st.session_state.sequence_log = []
                st.session_state.nmr.sequence_log = []
                st.session_state.current_sequence = []
                st.session_state.nmr.reset()
                st.success("All cleared!")
    
    # Plotting section
    st.header("Results")
    
    if hasattr(st.session_state.nmr, 'fid') and st.session_state.nmr.fid is not None:
        fig = st.session_state.nmr.plot_1D()
        st.pyplot(fig)
        
        # Final state display
        st.subheader("Final System State")
        
        # Calculate magnetization components
        Mx_A = st.session_state.nmr.gamma_A * np.trace(st.session_state.nmr.rho @ st.session_state.nmr.Ax)
        My_A = st.session_state.nmr.gamma_A * np.trace(st.session_state.nmr.rho @ st.session_state.nmr.Ay)
        Mz_A = st.session_state.nmr.gamma_A * np.trace(st.session_state.nmr.rho @ st.session_state.nmr.Az)
        
        Mx_K = st.session_state.nmr.gamma_K * np.trace(st.session_state.nmr.rho @ st.session_state.nmr.Kx)
        My_K = st.session_state.nmr.gamma_K * np.trace(st.session_state.nmr.rho @ st.session_state.nmr.Ky)
        Mz_K = st.session_state.nmr.gamma_K * np.trace(st.session_state.nmr.rho @ st.session_state.nmr.Kz)
        
        # Display magnetization in columns
        mag_col1, mag_col2 = st.columns(2)
        
        with mag_col1:
            st.write("**Spin A Magnetization:**")
            st.write(f"Mx = {Mx_A.real:.6f} + {Mx_A.imag:.6f}i")
            st.write(f"My = {My_A.real:.6f} + {My_A.imag:.6f}i")
            st.write(f"Mz = {Mz_A.real:.6f} + {Mz_A.imag:.6f}i")
            st.write(f"|M| = {np.sqrt(Mx_A.real**2 + My_A.real**2 + Mz_A.real**2):.6f}")
        
        with mag_col2:
            st.write("**Spin K Magnetization:**")
            st.write(f"Mx = {Mx_K.real:.6f} + {Mx_K.imag:.6f}i")
            st.write(f"My = {My_K.real:.6f} + {My_K.imag:.6f}i")
            st.write(f"Mz = {Mz_K.real:.6f} + {Mz_K.imag:.6f}i")
            st.write(f"|M| = {np.sqrt(Mx_K.real**2 + My_K.real**2 + Mz_K.real**2):.6f}")
        
        # Density matrix elements (show only significant ones)
        st.write("**Density Matrix Elements (significant only):**")
        
        # Find significant off-diagonal elements
        rho = st.session_state.nmr.rho
        significant_elements = []
        
        for i in range(4):
            for j in range(4):
                if i != j and abs(rho[i,j]) > 1e-6:
                    significant_elements.append((i, j, rho[i,j]))
        
        if significant_elements:
            for i, j, val in significant_elements:
                st.write(f"ρ[{i},{j}] = {val.real:.6f} + {val.imag:.6f}i")
        else:
            st.write("No significant off-diagonal elements (diagonal density matrix)")
        
        # Show all density matrix elements for debugging
        st.write("**Full Density Matrix (for debugging):**")
        for i in range(4):
            row = []
            for j in range(4):
                val = rho[i,j]
                if abs(val) < 1e-10:
                    row.append("0.000000")
                else:
                    row.append(f"{val.real:.6f}+{val.imag:.6f}i")
            st.write(f"Row {i}: {', '.join(row)}")
        
        # Diagonal elements (populations)
        st.write("**Populations (diagonal elements):**")
        for i in range(4):
            st.write(f"ρ[{i},{i}] = {rho[i,i].real:.6f} + {rho[i,i].imag:.6f}i")
        
        # Trace (should be 1)
        trace = np.trace(rho)
        st.write(f"**Trace: {trace.real:.6f} + {trace.imag:.6f}i**")
        
    else:
        st.info("No acquisition data available. Run an acquisition to see results.")
    

if __name__ == "__main__":
    main()
