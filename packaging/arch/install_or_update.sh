#!/usr/bin/env bash
# Build the committed `main` branch and install it, or upgrade the installed copy.
set -euo pipefail

cd "$(dirname "$(realpath "$0")")"
repo="$(realpath ../..)"

if [ -n "$(git -C "$repo" status --porcelain -- . ':!packaging')" ]; then
  echo "Note: uncommitted changes are not included; this builds the committed main branch."
fi

if pacman -Q music-looper &>/dev/null; then
  action="Updated"
else
  action="Installed"
fi

rm -f music-looper-*.pkg.tar.*
# -s installs missing dependencies, -i installs/upgrades the result, -f rebuilds, -c cleans up afterwards
makepkg -sifc

echo "$action $(pacman -Q music-looper)"
