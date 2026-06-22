#!/usr/bin/env bash
# Generate synthetic source media so the demo is fully self-contained.
set -euo pipefail
DIR="$(dirname "$0")/assets"
mkdir -p "$DIR"

# Two 10s "interview" source clips (different test patterns + spoken-ish tones)
ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "testsrc2=size=1280x720:rate=30:duration=10" \
  -f lavfi -i "sine=frequency=320:duration=10" \
  -pix_fmt yuv420p -c:v libx264 -crf 23 -c:a aac "$DIR/speaker_a.mp4"

ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "smptebars=size=1280x720:rate=30:duration=10" \
  -f lavfi -i "sine=frequency=440:duration=10" \
  -pix_fmt yuv420p -c:v libx264 -crf 23 -c:a aac "$DIR/speaker_b.mp4"

# Background music bed (quiet harmonic tone), 30s
ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "sine=frequency=110:duration=30" \
  -af "volume=0.25" -c:a aac "$DIR/music.m4a"

# A title card image
ffmpeg -y -hide_banner -loglevel error \
  -f lavfi -i "color=c=0x0B0B14:size=1920x1080:duration=1" \
  -frames:v 1 "$DIR/titlecard.png"

echo "assets written to $DIR"
ls -la "$DIR"
