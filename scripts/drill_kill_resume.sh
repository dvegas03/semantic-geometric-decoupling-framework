#!/usr/bin/env bash
# Gate drill: SIGTERM then SIGKILL mid-LoRA-step, verifying checkpoint/resume survives both.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DRILL_DIR="${DRILL_DIR:-$(mktemp -d)}"
QUEUE="${DRILL_DIR}/drill_queue.yaml"
export PILOT_ROOT="${DRILL_DIR}/pilot"
CKPT="${PILOT_ROOT}/ckpts/lora-drill"
mkdir -p "${PILOT_ROOT}"

cat > "${QUEUE}" <<'EOF'
experiments:
  - name: lora-drill
    kind: lora
    config: {max_steps: 120}
EOF

if [ ! -d "${PILOT_ROOT}/corpus" ] || [ -z "$(ls -A "${PILOT_ROOT}/corpus" 2>/dev/null || true)" ]; then
  python - <<'PY'
from pathlib import Path
import json, threading
from training.profile import Profile
from training.queue import ExperimentSpec
from training.checkpoint import CheckpointManager
from training.datagen import DatagenJob

profile = Profile.load("local")
profile = Profile.load("local")
from dataclasses import replace
corpus = replace(profile.corpus, frames=64, shard_size=32)
profile = replace(profile, corpus=corpus)
spec = ExperimentSpec(name="datagen-drill", kind="datagen", config={})
ck = CheckpointManager(profile.paths.ckpt_root / "datagen-drill")
job = DatagenJob(spec, profile, ck, threading.Event())
assert job.run().value == "completed"
print("drill corpus ready under", profile.paths.data_root)
PY
fi

latest_step() {
  if [ ! -f "${CKPT}/metrics.jsonl" ]; then echo 0; return; fi
  python - <<PY
import json
from pathlib import Path
p = Path("${CKPT}/metrics.jsonl")
step = 0
for line in p.read_text().splitlines():
    if line.strip():
        step = json.loads(line).get("step", step)
print(step)
PY
}

checkpoint_step() {
  if [ ! -f "${CKPT}/LATEST" ]; then echo 0; return; fi
  python - <<PY
from pathlib import Path
p = Path("${CKPT}/LATEST")
name = p.read_text().strip()
if name.startswith("step_"):
    print(int(name.split("_")[1]))
else:
    import torch
    state = torch.load(Path("${CKPT}") / name / "train_state.pt", map_location="cpu", weights_only=False)
    print(int(state.get("step", 0)))
PY
}

assert_queue_status() {
  local name="$1" want="$2"
  python - <<PY
import yaml
from pathlib import Path
q = yaml.safe_load(Path("${QUEUE}").read_text())
entry = next(e for e in q["experiments"] if e["name"] == "$name")
assert entry["status"] == "$want", entry
print("queue status ok:", entry["status"])
PY
}

assert_resume_log() {
  local needle="$1"
  grep -q "${needle}" "${DRILL_DIR}/runner.log" && echo "resume log ok"
}

run_until_step() {
  local target="$1"
  python -m training.runner --profile local --queue "${QUEUE}" \
    >"${DRILL_DIR}/runner.log" 2>&1 &
  RUNNER=$!
  for _ in $(seq 1 360); do
    cur="$(latest_step)"
    if [ "${cur}" -ge "${target}" ]; then
      return 0
    fi
    if ! kill -0 "${RUNNER}" 2>/dev/null; then
      echo "runner exited early; log:" >&2
      tail -50 "${DRILL_DIR}/runner.log" >&2
      return 1
    fi
    sleep 5
  done
  echo "timeout waiting for step ${target}" >&2
  return 1
}

DRILL_PROFILE="${DRILL_DIR}/drill_profile.yaml"
python - <<PY
from pathlib import Path
import yaml
raw = yaml.safe_load(Path("configs/profiles/local.yaml").read_text())
raw["checkpoint_interval_s"] = 15
raw["lora"]["max_steps"] = 120
Path("${DRILL_PROFILE}").write_text(yaml.safe_dump(raw))
PY

run_until_step_profile() {
  local target="$1"
  python -m training.runner --profile "${DRILL_PROFILE}" --queue "${QUEUE}" \
    >"${DRILL_DIR}/runner.log" 2>&1 &
  RUNNER=$!
  for _ in $(seq 1 720); do
    cur="$(latest_step)"
    if [ "${cur}" -ge "${target}" ]; then
      return 0
    fi
    if ! kill -0 "${RUNNER}" 2>/dev/null; then
      echo "runner exited early; log:" >&2
      tail -80 "${DRILL_DIR}/runner.log" >&2
      return 1
    fi
    sleep 5
  done
  echo "timeout waiting for step ${target}" >&2
  return 1
}

echo "=== Drill 1: SIGTERM ==="
run_until_step_profile 5
kill -TERM "${RUNNER}"
set +e
wait "${RUNNER}"
code=$?
set -e
[ "${code}" -eq 0 ]
assert_queue_status lora-drill pending
S1=$(checkpoint_step)
echo "checkpoint at step ${S1}"

echo "=== Drill 2: resume ==="
run_until_step_profile $((S1 + 3))
kill -TERM "${RUNNER}"
set +e
wait "${RUNNER}" || true
set -e
assert_resume_log "resumed from step ${S1}"

echo "=== Drill 3: SIGKILL ==="
run_until_step_profile $((S1 + 6))
kill -9 "${RUNNER}" || true
wait "${RUNNER}" 2>/dev/null || true
python - <<PY
import yaml
from pathlib import Path
path = Path("${QUEUE}")
raw = yaml.safe_load(path.read_text())
for e in raw["experiments"]:
    if e["name"] == "lora-drill":
        e["status"] = "pending"
        e.pop("owner", None)
        e.pop("heartbeat", None)
path.write_text(yaml.safe_dump(raw, sort_keys=False))
print("forced lora-drill back to pending after SIGKILL")
PY
python -m training.runner --profile "${DRILL_PROFILE}" --queue "${QUEUE}" \
  >>"${DRILL_DIR}/runner.log" 2>&1
assert_queue_status lora-drill done
echo "DRILL PASSED"
