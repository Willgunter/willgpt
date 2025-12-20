!pip install -q --force-reinstall \
    torch==2.7.1+cu128 \
    torchvision==0.22.1+cu128 \
    torchaudio==2.7.1+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

just make sure you set up env vars before you run:
ex:

cd /scratch/$SLURM_JOB_ID
mkdir -p run
cd run

export HF_HOME=$PWD/.hf
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers

Colab usage (examples):

python evals/eleuther-ai-evals/eleuther-ai-eval.py \
  --models google/gemma-2-2b \
  --benchmarks arc_challenge,gsm8k \
  --batch-size 2 \
  --model-args dtype=float16,device_map=auto \
  --output results.json

Notes:
- --batch-size is "how many prompts in parallel" per forward pass; larger is faster but uses more GPU memory.
- If you see CUDA out-of-memory (OOM), try --batch-size 1 (and keep dtype=float16).
