#!/bin/sh
set -e

SAMPLE_DIR="${SAMPLE_DATA_DIR:-/app/sample_data}"

need_generate=0
for ep in EP_0038 EP_0039 EP_0040 EP_0041 EP_0042; do
  if [ ! -f "$SAMPLE_DIR/episodes/$ep/metadata.json" ]; then
    need_generate=1
    break
  fi
done

if [ "$need_generate" -eq 1 ]; then
  echo "[entrypoint] Generating all synthetic episodes…"
  python /app/scripts/generate_episode.py --out "$SAMPLE_DIR" --all
else
  echo "[entrypoint] Sample data ready under $SAMPLE_DIR/episodes"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
