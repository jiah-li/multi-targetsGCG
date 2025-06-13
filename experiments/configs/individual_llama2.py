import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_llama2'

    config.tokenizer_paths=["/data/jiahli/mogcg/llm-attacks/DIR/llama-2-7b-chat-hf"]
    config.model_paths=["/data/jiahli/mogcg/llm-attacks/DIR/llama-2-7b-chat-hf"]
    config.conversation_templates=['llama-2']

    return config

# huggingface-cli download --token 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 --resume-download meta-llama/llama-2-7b-chat-hf --local-dir /data/jiahli/mogcg/llm-attacks/DIR/llama-2-7b-chat-hf