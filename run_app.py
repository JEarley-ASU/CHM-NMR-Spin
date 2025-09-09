#!/usr/bin/env python3
"""
Simple script to run the NMR Spin System Simulator webapp
"""

import subprocess
import sys
import os

def main():
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("Streamlit not found. Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Run the app
    print("Starting NMR Spin System Simulator...")
    print("The app will open in your default web browser.")
    print("Press Ctrl+C to stop the server.")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\nShutting down server...")

if __name__ == "__main__":
    main()
