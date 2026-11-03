#!/usr/bin/env bash
# Parse PDFs with MinerU into the layout bioparser-build-dataset expects.
#
#   scripts/parse_pdfs.sh [PDF_DIR] [OUT_DIR]      (defaults: fine_tuning_data/pdfs, fine_tuning_data/parsed)
#
# * MinerU runs in an isolated, version-pinned uv tool environment and never touches the
#   training venv (its dependencies would conflict with the Unsloth stack).
# * MINERU_VERSION must equal the version used in production. Otherwise the model is trained
#   on one markdown dialect and sees another at inference.
# * Output: one <pdf-stem>.zip per PDF with markdown.md, middle_json.json,
#   structured_content.json and images (verified for mineru-kit 4.1.6). Name PDFs like the
#   TSVs without the "ETS_" prefix so they pair automatically.
# * Local parsing only. Do NOT add --remote: it uploads the client's PDFs to mineru.net.
set -euo pipefail

MINERU_VERSION="${MINERU_VERSION:-4.1.6}"
TIER="${TIER:-basic}" # basic = local lightweight models; standard/advanced use a VLM
PDF_DIR="${1:-fine_tuning_data/pdfs}"
OUT_DIR="${2:-fine_tuning_data/parsed}"

mkdir -p "$OUT_DIR"
cmd=(uvx --from "mineru==${MINERU_VERSION}" mineru-kit parse "$PDF_DIR" -o "$OUT_DIR" --format zip --tier "$TIER")
if command -v systemd-inhibit >/dev/null 2>&1; then # keep the machine awake for long batches
  cmd=(systemd-inhibit --what=idle --who=mineru --why="PDF parsing" "${cmd[@]}")
fi
"${cmd[@]}"
echo "Parsed into $OUT_DIR. Next: uv run bioparser-build-dataset"
