import torch
from torch import nn
import model
import data_loader
from absl import flags
from absl import app

# Hyperparameters similar to GPT-3 Small.
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 32,
    'Batch of independent sequences to train or process in parallel.')
_BLOCK_SIZE = flags.DEFINE_integer('block_size', 8,
                                   'Maximum context length for prediction.')
_MAX_ITERS = flags.DEFINE_integer('max_iters', 8000,
                                  'Maximum number of training iterations.')
_EVAL_INTERVAL = flags.DEFINE_integer(
    'eval_interval', 500, 'Number of training iterations between evaluations.')
_EVAL_ITERS = flags.DEFINE_integer('eval_iters', 200,
                                   'Number of iterations to evaluate.')
_LEARNING_RATE = flags.DEFINE_float('learning_rate', 6e-4, 'Learning rate.')
_EMBEDDING_SIZE = flags.DEFINE_integer('embedding_size', 768, 'Embedding size.')
_ATTENTION_LAYERS = flags.DEFINE_integer('attention_layers', 12,
                                         'Number of attention layers.')
_ATTENTION_HEADS = flags.DEFINE_integer('attention_heads', 12,
                                        'Number of attention heads.')
_DROPOUT = flags.DEFINE_float('dropout', 0.2, 'Dropout rate.')


@torch.no_grad()
def estimate_loss(decoder: nn.Module,
                  dataset: data_loader.DataLoader) -> tuple[float, float]:
  '''Returns the train and eval losses.'''
  decoder.eval()
  out = {}
  for split in ['train', 'val']:
    losses = torch.zeros(_EVAL_ITERS.value)
    for k in range(_EVAL_ITERS.value):
      X, Y = dataset.get_batch(split)
      _, loss = decoder(X, Y)
      losses[k] = loss.item()
    out[split] = losses.mean()
  decoder.train()
  return out['train'], out['val']


def main(_):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  dataset = data_loader.DataLoader(_BATCH_SIZE.value,
                                   _BLOCK_SIZE.value,
                                   data_path='data/test.txt',
                                   device=device)

  decoder = model.Decoder(num_attention_layers=_ATTENTION_LAYERS.value,
                          num_attention_heads=_ATTENTION_HEADS.value,
                          embedding_size=_EMBEDDING_SIZE.value,
                          block_size=_BLOCK_SIZE.value,
                          vocab_size=len(dataset.get_vocabs()),
                          dropout=_DROPOUT.value).to(device)

  print(sum(p.numel() for p in decoder.parameters()) / 1e6, 'M parameters')

  optimizer = torch.optim.AdamW(decoder.parameters(), lr=_LEARNING_RATE.value)

  for iter in range(_MAX_ITERS.value):

    # Every once in a while evaluate the loss on train and val sets.
    if iter % _EVAL_INTERVAL.value == 0:
      train_loss, val_loss = estimate_loss(decoder=decoder, dataset=dataset)
      print(
          f"step {iter}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    x, y = dataset.get_batch('train')
    _, loss = decoder(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

  # Generate from the model.
  prefix = 'Hi,'
  context = torch.tensor(dataset.encode(prefix),
                         dtype=torch.long,
                         device=device)

  print(dataset.decode(decoder.complete(context, length=500).tolist()))


if __name__ == '__main__':
  app.run(main)
