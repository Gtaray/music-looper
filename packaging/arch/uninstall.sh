#!/usr/bin/env bash
# Remove the installed music-looper package.
set -euo pipefail

if pacman -Q music-looper &>/dev/null; then
  sudo pacman -R music-looper
  echo "Uninstalled music-looper."
else
  echo "music-looper is not installed."
fi
