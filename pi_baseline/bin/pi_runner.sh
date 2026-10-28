#!/usr/bin/env bash
set -euo pipefail

# Determine script root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Ensure PATH includes Node / Homebrew
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Enforce isolated Pi configuration directory
export PI_CONFIG_DIR="$BASELINE_ROOT/.pi/agent"

# Point to the local isolated binary
PI_BIN="$BASELINE_ROOT/node_modules/.bin/pi"

if [ ! -f "$PI_BIN" ]; then
    echo "ERROR: Pi binary not found at $PI_BIN. Please run 'npm install' inside $BASELINE_ROOT." >&2
    exit 1
fi

# Execute the isolated Pi agent
exec "$PI_BIN" "$@"
