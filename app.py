import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import io
import pandas as pd

def safe_round(value, decimals=8):
    """Safely round complex numbers to avoid overflow errors"""
    if np.iscomplexobj(value):
        return np.round(value.real, decimals) + 1j * np.round(value.imag, decimals)
    else:
        return np.round(value, decimals)

class SpinSystem:
    def __init__(self, delta_A=10.0, delta_K=25.0, J=0.0, T2=0.5,
                 gamma_A=1.0, gamma_K=1):  # A=1H (ref), K=13C (~0.251)
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
        self.rho = (pA*self.Az + pK*self.Kz) / 2
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

        # Apply rotation
        self.rho = R @ self.rho @ R.conj().T
        # Round to prevent overflow errors
        self.rho = safe_round(self.rho)

        # Debug: check magnetization after pulse
        Mx_A = safe_round(self.gamma_A * np.trace(self.rho @ self.Ax))
        My_A = safe_round(self.gamma_A * np.trace(self.rho @ self.Ay))
        Mx_K = safe_round(self.gamma_K * np.trace(self.rho @ self.Kx))
        My_K = safe_round(self.gamma_K * np.trace(self.rho @ self.Ky))
        print(f"After pulse: Mx_A={Mx_A:.6f}, My_A={My_A:.6f}, Mx_K={Mx_K:.6f}, My_K={My_K:.6f}")

        # Log the pulse
        phase_str = phase if isinstance(phase, str) else f"{np.degrees(phi):.0f}°"
        self.sequence_log.append(f"Pulse: {flip_angle}° {phase_str} on spin {spin}")
        
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
                # Round to prevent overflow errors
                self.rho = safe_round(self.rho)
                
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
                    # Round to prevent overflow errors
                    self.rho = safe_round(self.rho)
                    
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
                Mx = safe_round(self.gamma_A * np.trace(self.rho @ self.Ax))
                My = safe_round(self.gamma_A * np.trace(self.rho @ self.Ay))
            elif observe == 'K':
                Mx = safe_round(self.gamma_K * np.trace(self.rho @ self.Kx))
                My = safe_round(self.gamma_K * np.trace(self.rho @ self.Ky))
            else:  # 'both'
                Mx = safe_round(self.gamma_A * np.trace(self.rho @ self.Ax) +
                      self.gamma_K * np.trace(self.rho @ self.Kx))
                My = safe_round(self.gamma_A * np.trace(self.rho @ self.Ay) +
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
                # Round to prevent overflow errors
                self.rho = safe_round(self.rho)
                
                # T2 decay
                decay = np.exp(-ti / self.T2)
                for m in range(4):
                    for n in range(4):
                        if m != n:
                            self.rho[m, n] *= decay
            
            # Detect magnetization
            if observe == 'K':
                Mx = safe_round(self.gamma_K * np.trace(self.rho @ self.Kx))
                My = safe_round(self.gamma_K * np.trace(self.rho @ self.Ky))
            elif observe == 'A':
                Mx = safe_round(self.gamma_A * np.trace(self.rho @ self.Ax))
                My = safe_round(self.gamma_A * np.trace(self.rho @ self.Ay))
            else:
                Mx = safe_round(self.gamma_A * np.trace(self.rho @ self.Ax) +
                      self.gamma_K * np.trace(self.rho @ self.Kx))
                My = safe_round(self.gamma_A * np.trace(self.rho @ self.Ay) +
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
        
        # Dynamic frequency range based on actual spectrum data
        # Find frequencies where spectrum has significant intensity
        spec_magnitude = np.abs(spec)
        max_intensity = np.max(spec_magnitude)
        
        if max_intensity > 0:
            # Find frequencies where intensity is above 1% of maximum
            threshold = 0.01 * max_intensity
            significant_indices = np.where(spec_magnitude > threshold)[0]
            
            if len(significant_indices) > 0:
                # Get frequency range of significant peaks
                min_freq = freq[significant_indices[0]]
                max_freq = freq[significant_indices[-1]]
                
                # Add 20% buffer on each side
                range_size = max_freq - min_freq
                buffer = max(range_size * 0.2, 5)  # Minimum 5 Hz buffer
                ax2.set_xlim([min_freq - buffer, max_freq + buffer])
            else:
                # Fallback: show center ± 50 Hz if no significant peaks
                ax2.set_xlim([-50, 50])
        else:
            # Fallback: show center ± 50 Hz if no signal
            ax2.set_xlim([-50, 50])
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
        
        # Gyromagnetic ratio (as a ratio slider)
        gamma_ratio = st.slider("γ_A : γ_K Ratio", 0.0, 1.0, 0.5, 0.01, 
                               help="Ratio of gyromagnetic ratios. Left = 4:1 (A:K), Middle = 1:1 (A:K), Right = 1:4 (A:K)")
        
        # Calculate individual gamma values from slider position
        # Position 0: A=4, K=1 (4:1 ratio)
        # Position 0.5: A=1, K=1 (1:1 ratio) 
        # Position 1: A=1, K=4 (1:4 ratio)
        if gamma_ratio <= 0.5:
            # Linear interpolation from 4:1 to 1:1
            # At 0: A=4, K=1. At 0.5: A=1, K=1
            gamma_A = safe_round(4.0 - 6.0 * gamma_ratio)  # 0->4, 0.5->1
            gamma_K = 1.0
        else:
            # Linear interpolation from 1:1 to 1:4
            # At 0.5: A=1, K=1. At 1: A=1, K=4
            gamma_A = 1.0
            gamma_K = safe_round(1.0 + 6.0 * (gamma_ratio - 0.5))  # 0.5->1, 1->4
        
        # Display the actual gamma values
        st.write(f"**γ_A = {gamma_A:.3f}**")
        st.write(f"**γ_K = {gamma_K:.3f}**")
        
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
        
        # Initial and Final state density matrix tables
        st.subheader("Density Matrix Analysis")
        
        # Get the current density matrix
        rho = st.session_state.nmr.rho
        
        # Create initial density matrix that matches the reset() method exactly
        # This should match: self.rho = (pA*self.Az + pK*self.Kz) / 2
        gamma_A = st.session_state.nmr.gamma_A
        gamma_K = st.session_state.nmr.gamma_K
        
        # Get the spin operators from the current system
        Az = st.session_state.nmr.Az
        Kz = st.session_state.nmr.Kz
        
        # Calculate initial density matrix exactly as in reset() method
        initial_rho = (gamma_A * Az + gamma_K * Kz) / 2
        
        # Display initial density matrix
        st.write("**Initial Density Matrix (Thermal Equilibrium):**")
        
        # Create column headers
        col_headers = ["", "|00⟩", "|01⟩", "|10⟩", "|11⟩"]
        
        # Create data rows for initial matrix
        initial_table_data = []
        for i in range(4):
            row = [f"⟨{i:02b}|"]  # Binary representation of row index
            for j in range(4):
                val = initial_rho[i, j]
                if abs(val) < 1e-10:
                    row.append("0")
                else:
                    # Format complex numbers nicely
                    if abs(val.imag) < 1e-10:
                        row.append(f"{val.real:.6f}")
                    else:
                        row.append(f"{val.real:.6f}{val.imag:+.6f}i")
            initial_table_data.append(row)
        
        # Display the initial table
        st.table(pd.DataFrame(initial_table_data, columns=col_headers))
        
        # Display final density matrix
        st.write("**Final Density Matrix (After Sequence):**")
        
        # Create data rows for final matrix
        final_table_data = []
        for i in range(4):
            row = [f"⟨{i:02b}|"]  # Binary representation of row index
            for j in range(4):
                val = rho[i, j]
                if abs(val) < 1e-10:
                    row.append("0")
                else:
                    # Format complex numbers nicely
                    if abs(val.imag) < 1e-10:
                        row.append(f"{val.real:.6f}")
                    else:
                        row.append(f"{val.real:.6f}{val.imag:+.6f}i")
            final_table_data.append(row)
        
        # Display the final table
        st.table(pd.DataFrame(final_table_data, columns=col_headers))
        
        # Add some additional information
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Trace
            trace = np.trace(rho)
            st.metric("Trace", f"{trace.real:.6f}{trace.imag:+.6f}i")
        
        with col2:
            # Purity (Tr(ρ²))
            purity = np.trace(rho @ rho)
            st.metric("Purity", f"{purity.real:.6f}")
        
        with col3:
            # Max off-diagonal element
            max_off_diag = 0
            for i in range(4):
                for j in range(4):
                    if i != j:
                        max_off_diag = max(max_off_diag, abs(rho[i, j]))
            st.metric("Max Coherence", f"{max_off_diag:.6f}")
        
        # Magnetization summary
        st.subheader("Magnetization Summary")
        
        # Calculate magnetization components
        Mx_A = safe_round(st.session_state.nmr.gamma_A * np.trace(rho @ st.session_state.nmr.Ax))
        My_A = safe_round(st.session_state.nmr.gamma_A * np.trace(rho @ st.session_state.nmr.Ay))
        Mz_A = safe_round(st.session_state.nmr.gamma_A * np.trace(rho @ st.session_state.nmr.Az))
        
        Mx_K = safe_round(st.session_state.nmr.gamma_K * np.trace(rho @ st.session_state.nmr.Kx))
        My_K = safe_round(st.session_state.nmr.gamma_K * np.trace(rho @ st.session_state.nmr.Ky))
        Mz_K = safe_round(st.session_state.nmr.gamma_K * np.trace(rho @ st.session_state.nmr.Kz))
        
        # Display in columns
        mag_col1, mag_col2 = st.columns(2)
        
        with mag_col1:
            st.write("**Spin A:**")
            st.write(f"Mx = {Mx_A.real:.6f} + {Mx_A.imag:.6f}i")
            st.write(f"My = {My_A.real:.6f} + {My_A.imag:.6f}i")
            st.write(f"Mz = {Mz_A.real:.6f} + {Mz_A.imag:.6f}i")
            st.write(f"|M| = {np.sqrt(Mx_A.real**2 + My_A.real**2 + Mz_A.real**2):.6f}")
        
        with mag_col2:
            st.write("**Spin K:**")
            st.write(f"Mx = {Mx_K.real:.6f} + {Mx_K.imag:.6f}i")
            st.write(f"My = {My_K.real:.6f} + {My_K.imag:.6f}i")
            st.write(f"Mz = {Mz_K.real:.6f} + {Mz_K.imag:.6f}i")
            st.write(f"|M| = {np.sqrt(Mx_K.real**2 + My_K.real**2 + Mz_K.real**2):.6f}")
        
    else:
        st.info("No acquisition data available. Run an acquisition to see results.")
    

if __name__ == "__main__":
    main()
