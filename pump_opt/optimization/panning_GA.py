import sko
import numpy as np
try:
    import numba
    from numba import njit, int32
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(f=None, *args, **kwargs):
        if f is None:
            return lambda fn: fn
        return f
from sko.tool_kit import x2gray
from scipy.ndimage import shift


class Panning_GA(sko.GA.GA):
  def __init__(
      self,
      func,
      n_dim,
      size_pop=50,
      max_iter=200,
      prob_mut=0.001,
      lb=-1,
      ub=1,
      constraint_eq=tuple(),
      constraint_ueq=tuple(),
      precision=1e-7,
      early_stop=None,
      panning_range=tuple(),
      paning_step=5,
      panning_prob=0.05,
      n_len=None,
  ):
    super().__init__(
        func, n_dim, size_pop, max_iter, prob_mut, lb, ub, constraint_eq,
        constraint_ueq, precision, early_stop
    )
    if n_len is None:
      self.n_len = np.array([n_dim])
    else:
      self.n_len = np.array(n_len)
    self.panning_range = panning_range
    self.paning_step = paning_step
    self.panning_prob = panning_prob

  def mutation(self):
    mask = np.random.rand(self.size_pop, self.len_chrom) < self.prob_mut
    self.Chrom ^= mask

    panning = np.random.randint(
        self.paning_step * 2 + 1, size=(self.size_pop, )
    ) - self.paning_step
    mask = np.random.rand(self.size_pop) > self.panning_prob
    panning[mask] = 0

    self.Chrom = roll(
        self.n_len, self.Chrom.astype(float), panning.astype(float),
        self.Lind.astype(float)
    ).astype(int)

  def crossover(self):
    mask = np.random.rand(self.size_pop, len(self.Lind)) < self.panning_prob
    mask = mask.astype(int)
    mask = np.repeat(mask, self.Lind, axis=1)
    chrom_up = shift(self.Chrom, [0, self.Lind[0]], cval=0)
    chrom_down = shift(self.Chrom, [0, -self.Lind[-1]], cval=0)
    mask_up = (chrom_up ^ self.Chrom) & mask
    chrom_up = self.Chrom ^ mask_up
    maks_down = (chrom_down ^ self.Chrom) & mask
    chrom_down = self.Chrom ^ maks_down
    mask2 = np.random.choice(
        self.size_pop, np.random.randint(self.size_pop), replace=False
    )
    chrom_down[mask2] = chrom_up[mask2]
    self.Chrom = chrom_down
    super().crossover()
    return self.Chrom

  def x2chrom(self, x, noise=1):
    """
    :param x: 1d array
    :param noise: float
    :return: 2d array
    """
    x = np.array(x)
    if x.shape == (1, self.n_dim):
      pass
    elif x.shape == (self.n_dim):
      x = x.reshape(1, self.n_dim)
    elif x.shape[-1] == self.n_dim and len(x.shape) == 2:
      x = x[np.random.choice(x.shape[0], self.size_pop)]
    elif not x:
      x = np.random.random([self.size_pop, self.n_dim])*(self.lb-self.ub)+self.ub
    else:
      raise ValueError('x shape error')
    noise = (np.random.random([self.size_pop, self.n_dim]) - 0.5) * noise * 2
    # x *= 1.4
    x = noise + x
    x = np.clip(x, self.lb, self.ub)
    chrom = np.stack(x2gray(x, self.n_dim, self.lb, self.ub,
                            self.precision)).astype(int)
    # self.Chrom = chrom
    return chrom


@njit
def roll(
    n_list=np.array([]),
    A=np.array([[]], ),
    r=np.array([], ),
    Lind=np.array([], )
):
  bias = 0
  lind_bias = 0
  for idx, n_len in enumerate(n_list):
    n_len = int(n_len)
    col_num = int(sum(Lind[lind_bias:lind_bias + n_len]))
    A_p = A[:, bias:bias + col_num]
    r_p = np.zeros_like(r)

    for idx, step in enumerate(r):
      step = int(step)
      if step > 0:
        step = int(sum(Lind[lind_bias:lind_bias + step]))
      elif step < 0:
        step = -int(sum(Lind[step + lind_bias + n_len:lind_bias + n_len]))
      r_p[idx] = step

    range_mat = np.random.randint(0, A_p.shape[1] // n_len, (A_p.shape[0], 2))
    range_mat *= n_len
    roll_lenth = np.abs(range_mat[:, 1] - range_mat[:, 0])
    r_p = np.remainder(r_p, roll_lenth).astype(np.intp)

    for idx in range(A_p.shape[0]):
      a, b = int(range_mat[idx, 0]), int(range_mat[idx, 1])
      if a > b:
        a, b = b, a
      elif a == b:
        continue

      roll_slice = A_p[idx][a:b]
      roll_slice = np.roll(roll_slice, -int(r_p[idx]))

      A_p[idx][a:b] = roll_slice

    A[:, bias:bias + col_num] = A_p
    bias += col_num
    lind_bias += n_len
  return A