#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Setting up System Packages and Environment ---"

# --- System Package Installation ---
echo "Updating package list and installing system dependencies..."
apt-get update && \
apt-get install -y libgl1 git gcc curl
echo "System dependencies installed."

# --- Tool Setup ---
TOOLS_DIR="/workspace/projects/threat-detector/tools"
mkdir -p "$TOOLS_DIR"
echo "Ensured tools directory exists: $TOOLS_DIR"

# --- Download and Setup websocat ---
WEBSOCAT_URL="https://github.com/vi/websocat/releases/download/v1.14.0/websocat.x86_64-unknown-linux-musl"
WEBSOCAT_BIN="$TOOLS_DIR/websocat"
if [ ! -f "$WEBSOCAT_BIN" ]; then
    echo "Downloading websocat..."
    # Use curl since it's installed above, redirect output to file
    curl -L -o "$WEBSOCAT_BIN" "$WEBSOCAT_URL"
    echo "Making websocat executable..."
    chmod +x "$WEBSOCAT_BIN"
    echo "websocat setup complete."
else
    echo "websocat already exists."
    chmod +x "$WEBSOCAT_BIN" # Ensure executable
fi

# --- Download and Setup ngrok ---
NGROK_URL="https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
NGROK_TGZ="$TOOLS_DIR/ngrok.tgz"
NGROK_BIN="$TOOLS_DIR/ngrok"
if [ ! -f "$NGROK_BIN" ]; then
    echo "Downloading ngrok..."
    # Use curl since it's installed above
    curl -L -o "$NGROK_TGZ" "$NGROK_URL"
    echo "Extracting ngrok..."
    tar xvzf "$NGROK_TGZ" -C "$TOOLS_DIR"
    if [ ! -f "$NGROK_BIN" ]; then
        echo "ERROR: ngrok binary not found after extraction!"
        exit 1
    fi
    rm "$NGROK_TGZ" # Clean up archive
    echo "ngrok setup complete."
else
    echo "ngrok already exists."
fi

# --- Configure ngrok Authtoken ---
NGROK_AUTHTOKEN="2h69umOhoYX5rhc7APMeYeBgYgN_7vqNdcGei3soUGHkm6Jrw"
echo "Adding ngrok authtoken..."
"$NGROK_BIN" config add-authtoken "$NGROK_AUTHTOKEN"
echo "ngrok authtoken added."

# --- Activate Conda Environment ---
# Check if Conda profile script exists
CONDA_PROFILE="/workspace/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_PROFILE" ]; then
    echo "Sourcing Conda profile..."
    # Using '.' or 'source' - '.' is more POSIX compliant
    . "$CONDA_PROFILE"
    echo "Activating Conda environment 'env_cc'..."
    conda activate env_cc
    echo "Conda environment 'env_cc' activated."
else
    echo "WARNING: Conda profile script not found at $CONDA_PROFILE. Cannot activate environment."
    # Decide if this is a fatal error or not
    # exit 1
fi

# --- Other Setup Steps (Add any other specific commands here) ---
echo "Running other setup steps (if any)..."
# Example: cd /workspace/projects/threat-detector/backend && pip install -r requirements.txt # If not handled elsewhere


# --- Reminder for Running Tools ---
echo "--- Setup Complete ---"
echo "Environment 'env_cc' should be active."
echo ""
echo "Remember to run ngrok and websocat commands as needed, likely in separate terminals."
echo "Use the full path: $TOOLS_DIR/ngrok or $TOOLS_DIR/websocat"
echo ""
echo "Example ngrok command (in a new terminal after running this script):"
echo "# $TOOLS_DIR/ngrok http 5001"
echo "# (You might need to add your auth token first: $TOOLS_DIR/ngrok config add-authtoken YOUR_TOKEN)"
echo ""
echo "Example websocat command (in a new terminal after running this script):"
echo "# $TOOLS_DIR/websocat ws://localhost:5001/ws/video_feed"
echo "# OR use the ngrok URL provided by the ngrok command." 