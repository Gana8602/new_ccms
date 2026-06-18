#!/bin/zsh
set -euo pipefail

SOURCE_VIDEO="${1:-/home/sirisha/Ganapathi/crowd1.mp4}"
RTSP_URL="${2:-rtsp://127.0.0.1:8554/mystream1}"

exec ffmpeg \
  -hide_banner \
  -loglevel warning \
  -use_wallclock_as_timestamps 1 \
  -fflags +genpts \
  -re \
  -stream_loop -1 \
  -i "$SOURCE_VIDEO" \
  -map 0:v:0 \
  -vf "scale=1280:-2:flags=bicubic,fps=20" \
  -c:v libx264 \
  -preset ultrafast \
  -tune zerolatency \
  -profile:v main \
  -level:v 4.0 \
  -pix_fmt yuv420p \
  -b:v 4M \
  -maxrate 4M \
  -bufsize 8M \
  -g 20 \
  -keyint_min 20 \
  -sc_threshold 0 \
  -an \
  -rtsp_transport tcp \
  -f rtsp \
  "$RTSP_URL"
