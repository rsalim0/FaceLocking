#!/usr/bin/env python3
"""
Face Locking ONNX Demo Launcher
This script helps launch the enrollment and locking demos.
"""
import subprocess
import sys
from pathlib import Path

def main():
    project_root = Path(__file__).parent
    venv_python = project_root / "venv" / "bin" / "python"
    
    if not venv_python.exists():
        print("Error: Virtual environment not found. Please run setup first.")
        sys.exit(1)
    
    db_path = project_root / "data" / "db" / "face_db.npz"
    
    print("=" * 50)
    print("Face Locking ONNX - Demo Launcher")
    print("=" * 50)
    print()
    
    # Check if enrollment is needed
    if not db_path.exists():
        print("⚠️  No enrollment found. Starting enrollment first...")
        print()
        print("Instructions:")
        print("  - Position yourself in front of the camera")
        print("  - Auto-capture is ENABLED - it will capture 15 samples automatically")
        print("  - Press 'q' to quit after enrollment completes")
        print()
        print("Starting enrollment now...")
        
        try:
            subprocess.run([str(venv_python), "-m", "src.enroll", "User"], check=True)
        except KeyboardInterrupt:
            print("\nEnrollment cancelled.")
            sys.exit(0)
        except subprocess.CalledProcessError as e:
            print(f"\nEnrollment failed: {e}")
            sys.exit(1)
        
        if not db_path.exists():
            print("\n⚠️  Enrollment did not complete. Database not created.")
            sys.exit(1)
        
        print("\n✅ Enrollment completed successfully!")
        print()
    
    # Start locking demo
    print("Starting Face Locking & Action Detection Demo...")
    print()
    print("Controls:")
    print("  n/p : Next/Previous identity")
    print("  +/- : Adjust distance threshold")
    print("  q   : Quit")
    print()
    print("The system will lock onto the selected identity and track actions.")
    print("History logs will be saved to data/history/")
    print()
    print("Starting locking demo now...")
    
    try:
        subprocess.run([str(venv_python), "-m", "src.locking"], check=True)
    except KeyboardInterrupt:
        print("\nDemo stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\nDemo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
