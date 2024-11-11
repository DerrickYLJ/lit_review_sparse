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
