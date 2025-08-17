"""
Simple launcher script for the Streamlit application.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit application."""
    app_path = os.path.join("app", "streamlit_app.py")
    
    if not os.path.exists(app_path):
        print(f"Error: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Launching California House Price Predictor...")
    print("🌐 The app will open in your default web browser")
    print("⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()