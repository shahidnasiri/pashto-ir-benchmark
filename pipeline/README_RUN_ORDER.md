# Pashto IR Benchmark — Local Reproduction Pipeline

**Always run scripts from the project root**, not from inside `scripts/`:

```
cd "Your\project\path"
python scripts\01_encode_embeddings.py --model bge-m3
```

## 0. One-time setup

1. Open Anaconda Prompt, `cd` into `envs\`, run `01_create_envs.bat`.
   This builds three conda environments: `pashto-main`, `pashto-jina-gte`, `pashto-nomic`.


## 1. Encode embeddings (repeat once per model, in the matching env)

```
conda activate pashto-main
python scripts\01_encode_embeddings.py --model labse
python scripts\01_encode_embeddings.py --model paraphrase-mpnet
python scripts\01_encode_embeddings.py --model paraphrase-minilm
python scripts\01_encode_embeddings.py --model multilingual-e5-large-instruct
python scripts\01_encode_embeddings.py --model bge-m3
python scripts\01_encode_embeddings.py --model pashto-bert
conda deactivate

conda activate pashto-jina-gte
python scripts\01_encode_embeddings.py --model jina-embeddings-v3
python scripts\01_encode_embeddings.py --model gte-multilingual-base
conda deactivate

conda activate pashto-nomic
python scripts\01_encode_embeddings.py --model nomic-embed-text-v2-moe
python scripts\01_encode_embeddings.py --model qwen3-embedding-0.6b
conda deactivate
```

If you hit a CUDA out-of-memory error on Qwen3-Embedding-0.6B or Jina-v3, lower batch size:
`--batch_size 8` or even `--batch_size 4`.

**First run of each model** will download weights from Hugging Face
(a few hundred MB to ~2GB each) — this needs internet once; after that
it's cached locally (`~/.cache/huggingface`) and later runs are offline.

## 2. Evaluate retrieval (no GPU needed — any env works)

Run for **all 10 models** — this writes `results/table5_overall_effectiveness.csv`
and one `results/ranks_{model}.csv` per model, which every later script needs:

```
for %m in (nomic-embed-text-v2-moe gte-multilingual-base jina-embeddings-v3 labse paraphrase-mpnet paraphrase-minilm multilingual-e5-large-instruct bge-m3 qwen3-embedding-0.6b pashto-bert) do python scripts\02_evaluate_retrieval.py --model %m
```

## 3. Significance tests, stratified analysis, generator fingerprint

Only after step 2 has produced all 10 `ranks_*.csv` files:

```
python scripts\03_significance_tests.py
python scripts\04_stratified_analysis.py
python scripts\05_generator_fingerprint.py
```

## 4. Generate every figure

```
python scripts\06_generate_figures.py
```
