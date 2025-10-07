import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

class Qwen2ForCausalLMWithLMHead(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)

        if not hasattr(self, "lm_head"):
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)