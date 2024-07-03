# RAG

## Hardware and LLM
For hardware, RTX 4090 GPU is required.\
For LLM, https://huggingface.co/cognitivecomputations/dolphin-2.9.2-qwen2-7b is selected as models
## Set up LLM serving 
- Firstly, install vllm as llm serving engine by below command:
```bash
pip install vllm
```

- Run llm serving by below command:
```bash
python3 -m vllm.entrypoints.openai.api_server --model cognitivecomputations/dolphin-2.9.2-qwen2-7b --dtype auto --gpu-memory-utilization "0.98" --enforce-eager --max-model-len "4096" --port 8888
```

## Set up RAG application
- Set up environment for rag application:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
- Run application:
```bash
python3 main.py
```