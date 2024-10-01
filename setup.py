import subprocess
import sys

def install_requirements():
    try:
        import cv2
        import mediapipe
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

if __name__ == "__main__":
    install_requirements()
    import main  # Replace with ur main script
    main.run()  
