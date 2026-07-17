#!/usr/bin/env bash
# Vista login-node prefetch: stages models/containers under $SCRATCH before the first sbatch chain job.
set -uo pipefail

SCRATCH="${SCRATCH:-${HOME}/scratch}"
HF_HOME="${SCRATCH}/hf"
CONTAINER_DIR="${SCRATCH}/containers"
CONTAINER_SIF="${CONTAINER_DIR}/pytorch-arm64.sif"
NGC_IMAGE="${NGC_IMAGE:-docker://nvcr.io/nvidia/pytorch:24.10-py3-igpu}"

MODELS=(
    "Qwen/Qwen2-VL-7B-Instruct"
    "Qwen/Qwen2-VL-2B-Instruct"
)

mkdir -p "${HF_HOME}" "${CONTAINER_DIR}"

echo "[prefetch] HF_HOME=${HF_HOME}"
echo "[prefetch] CONTAINER_SIF=${CONTAINER_SIF}"

if [ "${SKIP_MODEL_PULL:-0}" -eq 1 ]; then
    echo "[prefetch] SKIP_MODEL_PULL=1 — skipping huggingface-cli downloads."
else
    export HF_HOME
    for model in "${MODELS[@]}"; do
        echo "[prefetch] downloading ${model} -> ${HF_HOME}"
        huggingface-cli download "${model}" --local-dir-use-symlinks False
    done
fi

if [ "${SKIP_CONTAINER_PULL:-0}" -eq 1 ]; then
    echo "[prefetch] SKIP_CONTAINER_PULL=1 — skipping NGC container pull."
else
    if [ -f "${CONTAINER_SIF}" ]; then
        echo "[prefetch] ${CONTAINER_SIF} already present — skipping pull."
    else
        echo "[prefetch] pulling ${NGC_IMAGE} -> ${CONTAINER_SIF}"
        apptainer pull "${CONTAINER_SIF}" "${NGC_IMAGE}"
    fi
fi

if [ "${SKIP_VERIFY:-0}" -eq 1 ]; then
    echo "[prefetch] SKIP_VERIFY=1 — skipping offline forward-pass check."
else
    echo "[prefetch] verifying offline forward pass (HF_HUB_OFFLINE=1)..."
    HF_HOME="${HF_HOME}" HF_HUB_OFFLINE=1 apptainer exec --nv "${CONTAINER_SIF}" \
        python -c "
import os
assert os.environ.get('HF_HUB_OFFLINE') == '1', 'offline contract not set'
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import torch

model_name = '${MODELS[1]}'
processor = AutoProcessor.from_pretrained(model_name)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map='cuda'
)
messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'ping'}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=4)
print('offline forward pass OK, generated', out.shape[-1], 'tokens')
"
fi

echo "[prefetch] done."
