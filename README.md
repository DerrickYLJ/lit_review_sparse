### Sparse Attention using Index Store

Environment Setup (python version = 3.8):

```
pip install torch==2.3.1+cu121 torchvision torchaudio
pip install transformers==4.44.0 accelerate datasets evaluate wandb scikit-learn scipy sentencepiece

python setup.py develop
```

Reproduce results on Needle-in-the-Haystack:

```
### models:
# gradientai/Llama-3-8B-Instruct-Gradient-1048k
# meta-llama/Meta-Llama-3-8B-Instruct
# lmsys/longchat-7b-v1.5-32k
# NousResearch/Yarn-Llama-2-7b-128k
# meta-llama/Llama-3.1-8B

python examples/needle_test/needle_test.py \
    --model_name gradientai/Llama-3-8B-Instruct-Gradient-1048k \
    --attn_type "index" \ # index for TidalDecode; quest for Quest; None for full weight
    --max_length 10000 \
    --min_length 2000 \
    --rounds 1 \
    --attn_type index \
    --output_path ./needle \
    --run_name Index_s \
    --top_k  512 \
    --correction_layer 9 \
    --sparse_layer_start 2 \
    --jobs 14-15
```

Reproduce results on Perplexity:

```
MODELPATH=gradientai/Llama-3-8B-Instruct-Gradient-1048k
OUTPUT_DIR=results/ppl/longchat/full
mkdir -p $OUTPUT_DIR

budget=4096

nohup python -u examples/ppl/run_ppl.py \
    --model_name_or_path $MODELPATH \
    --attn_type "index" \ # index for TidalDecode; quest for Quest; None for full weight
    --output_dir $OUTPUT_DIR \
    --num_eval_tokens 32000 \
    --correction_layer 9 \
    --sparse_layer_start 2 \
    --top_k $budget  --chunk_size 16 > output_32k_full.log
```

Attention Sinks in Transformers for endless fluent generation


```
python benchmark/perplexity.py --experiment attention_sinks

usage: perplexity.py [-h] [--experiment {attention_sinks,transformers,windowed}] [--model_name_or_path MODEL_NAME_OR_PATH] [--revision REVISION]
                     [--trust_remote_code] [--dataset_name DATASET_NAME] [--data_column DATA_COLUMN] [--task TASK] [--split {validation,test}]
                     [--num_tokens NUM_TOKENS] [--output_dir OUTPUT_DIR] [--window_size WINDOW_SIZE] [--attention_sink_size ATTENTION_SINK_SIZE]

options:
  -h, --help            show this help message and exit
  --experiment {attention_sinks,transformers,windowed}
  --model_name_or_path MODEL_NAME_OR_PATH
  --revision REVISION
  --trust_remote_code
  --dataset_name DATASET_NAME
  --data_column DATA_COLUMN
  --task TASK
  --split {validation,test}
  --num_tokens NUM_TOKENS
  --output_dir OUTPUT_DIR
  --window_size WINDOW_SIZE
  --attention_sink_size ATTENTION_SINK_SIZE

python benchmark/plot_perplexity.py --features perplexity latency --title "Log perplexity & latency of Llama 2 7B as a function of input lengths"

usage: plot_perplexity.py [-h] [--output_dir OUTPUT_DIR] [--features {perplexity,vram,latency} [{perplexity,vram,latency} ...]] [--title TITLE]
                          [--log_perplexity_limit LOG_PERPLEXITY_LIMIT] [--skip_first SKIP_FIRST]

options:
  -h, --help            show this help message and exit
  --output_dir OUTPUT_DIR
  --features {perplexity,vram,latency} [{perplexity,vram,latency} ...]
  --title TITLE
  --log_perplexity_limit LOG_PERPLEXITY_LIMIT
  --skip_first SKIP_FIRST
```

H2O Performance Experiments Implementation
```
cd flexgen
python flex_opt.py --gpu-batch-size 1 --overlap false --hh-ratio 0.1 --hh-all --model facebook/opt-6.7b


Run Experiments
See test suite in h2o_flexgen/benchmark/h2o/h2o_suite.py

```


