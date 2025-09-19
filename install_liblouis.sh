#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# --- Dependency Checks ---
echo "Checking for required dependencies..."

# Check for Git
if ! command_exists git; then
    echo "Error: git is not installed. Please install it first."
    exit 1
fi

# Check for build tools (common names)
if ! (command_exists autoconf && command_exists automake && command_exists libtool && command_exists make && command_exists gcc); then
    echo "Warning: Build tools (autoconf, automake, libtool, make, gcc) might be missing."
    echo "Please install them using your package manager (e.g., 'sudo apt-get install build-essential' on Debian/Ubuntu)."
fi

# Check for pip
if ! command_exists pip3; then
    echo "Error: pip3 is not installed. Please install python3-pip."
    exit 1
fi

# --- Installation Steps ---

# 1. Clone the Liblouis Repository if it doesn't exist
if [ ! -d "liblouis" ]; then
    echo "Cloning the Liblouis repository..."
    git clone https://github.com/liblouis/liblouis.git
fi
cd liblouis

# 2. Configure, Compile and Make
echo "Configuring, compiling, and installing Liblouis..."
./autogen.sh
./configure --enable-ucs4
make
sudo make install

# 3. Install Python module
echo "Installing the Python module..."
cd python
sudo pip3 install .

# 4. Update library cache and environment
echo "Updating shared library cache..."


# Add LD_LIBRARY_PATH to shell profile if not already there
BASH_PROFILE="$HOME/.bashrc"
ZSH_PROFILE="$HOME/.zshrc"
LD_PATH_LINE="export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"

if [ -f "$BASH_PROFILE" ] && ! grep -q "LD_LIBRARY_PATH=/usr/local/lib" "$BASH_PROFILE"; then
    echo "Adding LD_LIBRARY_PATH to $BASH_PROFILE"
    echo "$LD_PATH_LINE" >> "$BASH_PROFILE"
fi

if [ -f "$ZSH_PROFILE" ] && ! grep -q "LD_LIBRARY_PATH=/usr/local/lib" "$ZSH_PROFILE"; then
    echo "Adding LD_LIBRARY_PATH to $ZSH_PROFILE"
    echo "$LD_PATH_LINE" >> "$ZSH_PROFILE"
fi

sudo ldconfig
echo "Liblouis installation complete."
echo "Please restart your terminal or run 'source ~/.bashrc' or 'source ~/.zshrc' to apply the changes."