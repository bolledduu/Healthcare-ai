set -euo pipefail
mkdir -p runs logs

states=(
  Massachusetts
  California
  Texas
  Florida
  Ohio
)

for i in "${!states[@]}"; do
  state="${states[$i]}"
  run_id="run_$((i+1))"
  out_dir="runs/${run_id}"
  log_file="logs/${run_id}.log"
  seed="$((1001 + i))"

  ./run_synthea \
    -s "${seed}" \
    -p 5000 \
    "${state}" \
    --exporter.baseDirectory="${out_dir}" \
    --exporter.csv.export=true \
    --exporter.fhir.export=true \
    --exporter.symptoms.csv.export=true \
    --exporter.symptoms.mode=1 \
    --exporter.csv.append_mode=false \
    --exporter.csv.folder_per_run=false \
    --exporter.symptoms.csv.append_mode=false \
    --exporter.symptoms.csv.folder_per_run=false \
    2>&1 | tee "${log_file}"
done
