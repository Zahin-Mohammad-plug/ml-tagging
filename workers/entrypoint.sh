#!/bin/bash
# Entrypoint script to fix permissions on shared volumes

# Fix permissions on shared frames cache volume
if [ -d "/app/.cache/frames" ]; then
    # Fix permissions if we have root access
    if [ "$(id -u)" = "0" ]; then
        chown -R worker:worker /app/.cache/frames 2>/dev/null || true
        chmod -R 755 /app/.cache/frames 2>/dev/null || true
    fi
fi

# Fix permissions on transformers cache directory
if [ -d "/tmp/cache/transformers" ]; then
    # Fix permissions if we have root access
    if [ "$(id -u)" = "0" ]; then
        chown -R worker:worker /tmp/cache/transformers 2>/dev/null || true
        chmod -R 755 /tmp/cache/transformers 2>/dev/null || true
        # Ensure .no_exist directories can be created
        find /tmp/cache/transformers -type d -name ".no_exist" -exec chmod 755 {} \; 2>/dev/null || true
    fi
fi

# Ensure transformers cache directory exists with proper permissions
if [ "$(id -u)" = "0" ]; then
    mkdir -p /tmp/cache/transformers
    chown -R worker:worker /tmp/cache/transformers 2>/dev/null || true
    chmod -R 755 /tmp/cache/transformers 2>/dev/null || true
fi

# Switch to worker user if we're root
if [ "$(id -u)" = "0" ]; then
    exec runuser -u worker -- "$@"
else
    exec "$@"
fi

