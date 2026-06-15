#!/bin/zsh
set -euo pipefail

SOURCE_VIDEO="${1:-/Volumes/tridelMac/Projects/Dev/ccms/video5.mp4}"
RTSP_URL="${2:-rtsp://127.0.0.1:8554/mystream}"

exec ffmpeg \
  -hide_banner \
  -loglevel warning \
  -re \
  -stream_loop -1 \
  -i "$SOURCE_VIDEO" \
  -map 0:v:0 \
  -vf "scale=1280:-2:flags=lanczos,fps=20" \
  -c:v libx264 \
  -preset veryfast \
  -tune zerolatency \
  -profile:v main \
  -level:v 4.0 \
  -pix_fmt yuv420p \
  -b:v 4M \
  -maxrate 4M \
  -bufsize 8M \
  -g 40 \
  -keyint_min 40 \
  -sc_threshold 0 \
  -an \
  -rtsp_transport tcp \
  -f rtsp \
  "$RTSP_URL"
