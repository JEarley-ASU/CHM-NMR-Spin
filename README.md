# NMR Spin System Simulator

An interactive web application for simulating two-spin NMR systems with pulse sequences, built with Streamlit.

## Features

- **Interactive Parameter Control**: Adjust chemical shifts, J-coupling, relaxation times, and gyromagnetic ratios
- **Pulse Sequence Builder**: Apply RF pulses with customizable flip angles, phases, and spin selection
- **Delay Controls**: Add delays for free evolution
- **Acquisition Options**: Acquire FIDs with or without decoupling
- **Real-time Visualization**: View FID and spectrum plots
- **Sequence Logging**: Track all applied operations
- **Example Sequences**: Pre-built sequences like INEPT and HSQC

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Open your browser to the URL shown in the terminal (typically http://localhost:8501)

## Usage

1. **Set Parameters**: Use the sidebar to adjust NMR parameters like chemical shifts, J-coupling, and relaxation times
2. **Build Sequences**: Use the pulse controls to apply RF pulses and delays
3. **Acquire Data**: Set acquisition parameters and run acquisitions
4. **View Results**: See FID and spectrum plots in real-time
5. **Try Examples**: Use the example sequence buttons for common NMR experiments

## Original Code

This webapp is based on the `Full_Spin.py` file, maintaining all the original functionality while adding an interactive web interface.
