#!/bin/sh
# Fetch EnergyPlus (pinned) into .scratch/ and verify it runs.
# Idempotent: if the binary already exists and reports the pinned version,
# nothing is downloaded. The install is scratch (gitignored), never committed.
# Usage: sh scripts/fetch_energyplus.sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
SCRATCH="$ROOT/.scratch"
VERSION="26.2.0"
SHA="4bd7a1f26f"
TARBALL="EnergyPlus-$VERSION-$SHA-Linux-Ubuntu24.04-x86_64.tar.gz"
URL="https://github.com/NatLabRockies/EnergyPlus/releases/download/v$VERSION/$TARBALL"
DIR="EnergyPlus-$VERSION-$SHA-Linux-Ubuntu24.04-x86_64"
BIN="$SCRATCH/$DIR/energyplus-$VERSION"

if [ -x "$BIN" ] && "$BIN" --version 2>/dev/null | grep -q "$VERSION"; then
  echo "EnergyPlus $VERSION already present at $BIN"
  exit 0
fi
mkdir -p "$SCRATCH"
cd "$SCRATCH"
echo "Downloading $URL ..."
curl -sSL -o "$TARBALL" "$URL"
echo "Extracting ..."
tar --no-same-owner -xzf "$TARBALL"
rm -f "$TARBALL"
"$BIN" --version
echo "EnergyPlus $VERSION ready at $BIN"
