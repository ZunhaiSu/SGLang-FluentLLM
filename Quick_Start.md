# Install SGLang-FluentLLM

You can set up the SGLang-FluentLLM environment on H20/H800 through the following steps

```bash
git clone git@github.com:meituan-longcat/SGLang-FluentLLM.git

pip3 install uv==0.7.2
uv venv fluentllmenv --python 3.11 --seed

source fluentllmenv/bin/activate
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"

cd SGLang-FluentLLM
git checkout main
git submodule update --init

pip3 install -e "./python[cuda_sm90]" --no-cache-dir

sh clean_setup.sh sm90
```

# LongCat-Flash-Lite Deploy Example

After setting up the environment, you can deploy the LongCat-Flash-Lite model using the following command:

```bash
# BF16 Model
MODEL_PATH=meituan-longcat/LongCat-Flash-Lite

# FP8 Model
MODEL_PATH=meituan-longcat/LongCat-Flash-Lite-FP8

python3 -m sglang.launch_server \
    --model-path ${MODEL_PATH} \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --port 10000 \
    --host 0.0.0.0 \
    --low-latency-max-num-tokens-per-gpu 2048 \
    --moe-parallel-strategy ep \
    --max-running-requests 32 \
    --nprocs-per-node 8 \
    --enable-flashinfer-mla \
    --attention-backend flashmla \
    --attn-tp-size 8 \
    --dp-size 1 \
    --log-level info \
```

To accelerate Decode via speculative inference, you can add the following configuration. Currently, only --speculative-eagle-topk = 1 is supported:

```bash
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --speculative-algorithm NEXTN \
    --draft-model-path-use-base \
```
