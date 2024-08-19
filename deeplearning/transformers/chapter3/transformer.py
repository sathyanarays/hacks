from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "this is not a test"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)


print(inputs)

from torch import nn
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_ckpt)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

inputs_embeds = token_emb(inputs.input_ids)
import torch

from math import sqrt
import torch.nn.functional as F



def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    #print("### Attention")
    #print("### Q", query.shape)
    #print("### K", key.shape)
    #print("### V", value.shape)
    scores = torch.bmm(query, key.transpose(1,2)) / sqrt(dim_k)    
    weights = F.softmax(scores, dim=-1)
    #print("### weights", weights)
    res = torch.bmm(weights, value)
    #print("### res", res.shape)
    return res

class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        attn_outputs = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
        )

        return attn_outputs
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim, head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim = -1)
        x = self.output_linear(x)
        return x
    
multihead_attn = MultiHeadAttention(config)
attn_output = multihead_attn(inputs_embeds)
print(attn_output.shape)

print(config)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x
    
feed_forward = FeedForward(config)
ff_outputs = feed_forward(attn_output)
print(ff_outputs.shape)