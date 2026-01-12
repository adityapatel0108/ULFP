#!/bin/bash
# Script to help transfer files to server
# Usage: ./transfer_to_server.sh username@server:/destination/path

if [ $# -eq 0 ]; then
    echo "Usage: ./transfer_to_server.sh username@server:/destination/path"
    echo "Example: ./transfer_to_server.sh user@jupyter.server.edu:/home/user/projects/"
    exit 1
fi

DEST=$1

echo "Transferring Linearizer framework to server..."
echo "Destination: $DEST"
echo ""

# Files and directories to transfer
FILES=(
    "src/"
    "notebooks/"
    "scripts/"
    "config.yaml"
    "requirements.txt"
    "README.md"
    "SETUP_GUIDE.md"
    "IMPLEMENTATION_SUMMARY.md"
    "quick_start.py"
)

# Exclude unnecessary files
EXCLUDE=(
    "--exclude=*.pyc"
    "--exclude=__pycache__"
    "--exclude=.git"
    "--exclude=checkpoints/"
    "--exclude=results/"
    "--exclude=data/"
    "--exclude=.ipynb_checkpoints"
)

echo "Transferring files..."
rsync -avz --progress "${EXCLUDE[@]}" "${FILES[@]}" "$DEST/fast-unlearning-face-recognition/"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Transfer complete!"
    echo ""
    echo "Next steps on server:"
    echo "1. cd fast-unlearning-face-recognition"
    echo "2. Update config.yaml with your dataset paths"
    echo "3. Run: python quick_start.py"
    echo "4. Or open: notebooks/03_linearization.ipynb"
else
    echo ""
    echo "✗ Transfer failed. Please check:"
    echo "  - Server connection"
    echo "  - Destination path permissions"
    echo "  - rsync is installed"
fi
