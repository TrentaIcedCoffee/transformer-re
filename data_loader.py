import torch


class DataLoader:

  def __init__(self,
               batch_size: int,
               block_size: int,
               data_path: str,
               device='cpu'):
    self._batch_size = batch_size
    self._block_size = block_size
    self._device = device

    with open(data_path, 'r', encoding='utf-8') as f:
      text = f.read()
    self._vocabs = list(set(text))

    self._c_to_i = {c: i for i, c in enumerate(self._vocabs)}
    self._i_to_c = {i: c for i, c in enumerate(self._vocabs)}

    data = torch.tensor(self.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    self._train_data = data[:n]
    self._val_data = data[n:]

  def get_batch(self, split):
    data = self._train_data if split == 'train' else self._val_data
    starts = torch.randint(len(data) - self._block_size, (self._batch_size,))
    x = torch.stack([data[i:i + self._block_size] for i in starts])  # (B, T)
    y = torch.stack([data[i + 1:i + self._block_size + 1] for i in starts
                    ])  # (B, T)
    return x.to(self._device), y.to(self._device)

  def get_vocabs(self):
    return self._vocabs

  def encode(self, s: str) -> list[int]:
    return [self._c_to_i[c] for c in s]

  def decode(self, idxs: list[int]) -> str:
    return ''.join(self._i_to_c[i] for i in idxs)
