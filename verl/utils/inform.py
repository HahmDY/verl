from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from transformers.utils import ModelOutput
from transformers.models.llama.modeling_llama import can_return_tuple
from transformers.models.llama.modeling_llama import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Model, Qwen2PreTrainedModel

@dataclass
class InfoRMOutputWithPast(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    mu: Optional[torch.FloatTensor] = None
    logvar: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    

class InfoRM(Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # base model architecture
        self.num_labels = config.num_labels
        self.model = Qwen2Model(config)
        self.latent_dim = 128
        
        # InfoRM
        self.encode_head = nn.Linear(config.hidden_size, self.latent_dim * 2, bias=False)
        self.score = nn.Linear(self.latent_dim, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> ModelOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        
        ib_representation = self.encode_head(hidden_states) # dim: (batch_size, latent_dim * 2)
        mu, logvar = ib_representation.chunk(2, dim=-1)
        
        if self.training:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mu + std * eps
        else:
            z = mu
            
        # (B, T, 1) -> (B, T)
        token_values = self.score(z).squeeze(-1)

        dev = token_values.device
        if attention_mask is not None:
            am = attention_mask.to(dev)
            eos_idx = am.size(1) - 1 - am.long().flip(-1).argmax(dim=1)
        else:
            eos_idx = torch.full((token_values.size(0),), token_values.size(1) - 1,
                                device=dev, dtype=torch.long)

        eos_idx = eos_idx.to(dev, dtype=torch.long)
        B = token_values.size(0)

        ar = torch.arange(B, device=dev, dtype=torch.long)

        pooled_logits = token_values[ar, eos_idx]  # (B,)

        idx3 = eos_idx.view(B, 1, 1).expand(B, 1, mu.size(-1))  # (B, 1, D)
        mu_eos = mu.gather(1, idx3).squeeze(1)                  # (B, D)
        logvar_eos = logvar.gather(1, idx3).squeeze(1)          # (B, D)

        loss = None

        return InfoRMOutputWithPast(
            loss=loss,
            mu=mu_eos,
            logvar=logvar_eos,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )