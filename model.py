import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
  '''Masked attention head.
  
  Args:
    embedding_size: embedding size.
    head_size: dimension of attention head.
    block_size: block size.
    dropout: dropout rate.

  Shape:
    Input: (batch_size, block_size, embedding_size).
    Output: (batch_size, block_size, head_size).
  '''

  def __init__(self, embedding_size: int, head_size: int, block_size: int,
               dropout: float):
    super().__init__()
    self.q = nn.Linear(embedding_size, head_size, bias=False)
    self.k = nn.Linear(embedding_size, head_size, bias=False)
    self.v = nn.Linear(embedding_size, head_size, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x: torch.Tensor):
    B, T, E = x.shape
    k = self.k(x)  # (B,T,H)
    q = self.q(x)  # (B,T,H)

    weight = q @ k.transpose(-1, -2) / (k.shape[-1]**0.5)  # (B,T,T)
    weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    prob = F.softmax(weight, dim=-1)  # (B,T,T)
    prob = self.dropout(prob)  # (B,T,T)

    return prob @ self.v(x)  # (B,T,H)


class MultiHeadAttention(nn.Module):
  '''Multi-head attention.
  
  Args:
    num_attention_heads: number of attention heads.
    head_size: dimension of each attention head.
    block_size: block size.
    dropout: dropout rate.

  Shape:
    Input: (batch_size, block_size, embedding_size).
    Output: (batch_size, block_size, embedding_size).
  '''

  def __init__(self, num_attention_heads: int, head_size: int,
               embedding_size: int, block_size: int, dropout: float):
    super().__init__()
    self.heads = nn.ModuleList([
        AttentionHead(embedding_size=embedding_size,
                      head_size=head_size,
                      block_size=block_size,
                      dropout=dropout) for _ in range(num_attention_heads)
    ])
    self.linear = nn.Linear(num_attention_heads * head_size,
                            num_attention_heads * head_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x: torch.Tensor):
    x = torch.cat([h(x) for h in self.heads], dim=-1)
    x = self.linear(x)
    x = self.dropout(x)
    return x


class FeedForward(nn.Module):

  def __init__(self, embedding_size: int, dropout: float):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(embedding_size, 4 * embedding_size),
        nn.ReLU(),
        nn.Linear(4 * embedding_size, embedding_size),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)


class TransformerBlock(nn.Module):
  '''Transformer block. Communication followed by computation.'''

  def __init__(self, num_attention_heads: int, embedding_size: int,
               block_size: int, dropout: float):
    assert embedding_size % num_attention_heads == 0, 'embedding size must be divisible by num heads'
    super().__init__()
    self.multi_head_attention = MultiHeadAttention(
        num_attention_heads=num_attention_heads,
        head_size=embedding_size // num_attention_heads,
        embedding_size=embedding_size,
        block_size=block_size,
        dropout=dropout)
    self.feed_forward = FeedForward(embedding_size=embedding_size,
                                    dropout=dropout)
    self.layer_norm_1 = nn.LayerNorm(embedding_size)
    self.layer_norm_2 = nn.LayerNorm(embedding_size)

  def forward(self, x):
    '''Forward pass with residual connection.'''
    x = x + self.multi_head_attention(self.layer_norm_1(x))
    x = x + self.feed_forward(self.layer_norm_2(x))
    return x


class Decoder(nn.Module):
  '''Decoder.
  
  Args:
    num_attention_layers: number of attention layers.
    embedding_size: embedding size.
    vocab_size: vocabulary size.
    block_size: block size.
  '''

  def __init__(self, num_attention_layers: int, num_attention_heads: int,
               embedding_size: int, block_size: int, vocab_size: int,
               dropout: float):
    super().__init__()
    self.block_size = block_size

    self.token_embedding = nn.Embedding(vocab_size, embedding_size)
    self.position_embedding = nn.Embedding(block_size, embedding_size)
    self.attention_blocks = nn.Sequential(*[
        TransformerBlock(num_attention_heads=num_attention_heads,
                         embedding_size=embedding_size,
                         block_size=block_size,
                         dropout=dropout) for _ in range(num_attention_layers)
    ])
    self.layer_norm = nn.LayerNorm(embedding_size)
    self.linear = nn.Linear(embedding_size, vocab_size)

  def forward(self, x: torch.Tensor, y=None):
    '''Forward pass.
    
    Args:
      x: (batch_size, block_size).
      y: (batch_size, block_size).

    Returns:
      logits: (batch_size, block_size, vocab_size).
      loss: scalar.
    '''
    B, T = x.shape

    token_embed = self.token_embedding(x)  # (B,T,E)
    pos_embed = self.position_embedding(torch.arange(T))  # (T,E)
    x = token_embed + pos_embed  # (B,T,E)
    x = self.attention_blocks(x)  # (B,T,E)
    x = self.layer_norm(x)  # (B,T,E)
    logits = self.linear(x)  # (B,T,V)

    if y is None:
      loss = None
    else:
      B, T, V = logits.shape
      loss = F.cross_entropy(logits.view(B * T, V), y.view(B * T))

    return logits, loss

  def complete(self, prefix: torch.Tensor, length: int):
    '''Complete text.
    
    Args:
      prefix: (prefix_length), prefix_length could be 1 or larger.
    '''
    x = prefix.reshape(1, -1)  # (1, prefix_length)
    for _ in range(length):
      logits, _ = self.forward(x[:, -self.block_size:])  # (1, T, V)
      logits = logits[:, -1, :]  # (1, V)
      probs = F.softmax(logits, dim=-1)  # (1, V)
      x_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
      x = torch.cat((x, x_next), dim=1)  # (1, T+1)

    return x[0]
