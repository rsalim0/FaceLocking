#!/bin/bash
# Face Locking ONNX Demo Runner
# This script runs enrollment and then the locking demo

cd "$(dirname "$0")"
source venv/bin/activate

echo "=========================================="
echo "Face Locking ONNX - Demo Setup Complete!"
echo "=========================================="
echo ""
echo "Step 1: Enrollment (auto-capture enabled)"
echo "Position yourself in front of the camera."
echo "The system will automatically capture 15 samples."
echo "Press 'q' to quit after enrollment completes."
echo ""
read -p "Press Enter to start enrollment..."
python -m src.enroll User

echo ""
echo "Step 2: Face Locking Demo"
echo "Use 'n'/'p' to select identity, 'q' to quit"
echo ""
read -p "Press Enter to start the locking demo..."
python -m src.locking
