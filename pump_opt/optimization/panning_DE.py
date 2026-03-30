from functools import lru_cache

import sko
import numpy as np
from numba import njit


# @njit
def swap_segments(arr: np.array([])):
  zero_indices = np.where(arr < 5)[0]
  zero_segments = np.split(
      zero_indices,
      np.where(np.diff(zero_indices) != 1)[0] + 1
  )
  nonzero_indices = np.where(arr >= 5)[0]
  nonzero_segments = np.split(
      nonzero_indices,
      np.where(np.diff(nonzero_indices) != 1)[0] + 1
  )

  # segments = np.array(zero_segments + nonzero_segments)
  segments = zero_segments + nonzero_segments
  np.random.shuffle(segments)

  return arr[np.concatenate(segments)]


# @njit(parallel=True)
def roll(
    X: np.array([[]]),
    panning: np.float_,
    n_len: np.array([]),
    prob_mut: np.float_,
):
  """

  roll X by paning each row
  """
  size_pop, _ = X.shape
  X = X.copy()
  swap_roll_flag = np.random.random(size=(size_pop, 3 * len(n_len))) < prob_mut
  for i in range(size_pop):
    bias = 0
    pop_list = []
    for idx, n in enumerate(n_len):
      if swap_roll_flag[i, idx]:
        pop_list += [swap_segments(X[i, bias:bias + n])]
      else:
        pop_list += [X[i, bias:bias + n]]
      if swap_roll_flag[i, idx + len(n_len)]:
        pop_list[-1] = np.roll(
            pop_list[-1], np.random.randint(-panning, panning)
        )
      if swap_roll_flag[i, idx + 2 * len(n_len)]:
        t = pop_list[-1]
        arg = np.random.choice(n, int(n / 3), replace=False)
        arg = arg[arg > 0]
        arg = arg[arg < n - 1]
        t[arg] = (t[arg - 1] + t[arg + 1]) / 2
      bias += n
    X[i] = np.concatenate(pop_list)
  return X


class Panning_DE(sko.DE.DE):
  def __init__(
      self,
      func,
      n_dim,
      F_num=1.,
      F_min=0.3,
      F_func=None,
      size_pop=50,
      max_iter=200,
      prob_mut=0.3,
      lb=-1,
      ub=1,
      constraint_eq=tuple(),
      constraint_ueq=tuple(),
      panning_range=tuple(),
      paning_step=3,
      panning_prob=0.05,
      CR=0.1,
      n_len=None,
      early_stop=50,
  ):
    super().__init__(
        func, n_dim, F_num, size_pop, max_iter, prob_mut, lb, ub, constraint_eq,
        constraint_ueq
    )
    self.F_func = F_func
    self.F_num = F_num
    self.panning_range = panning_range
    self.paning_step = paning_step
    self.panning_prob = panning_prob
    self.CR = CR
    self.F_min = F_min
    self.F_adp = (F_min / F_num)**(1 / max_iter)
    if n_len is None:
      self.n_len = np.array([n_dim])
    else:
      self.n_len = np.array(n_len)
    self.early_stop = early_stop

  def adaptive_random_F(self, F, CR, pop, fitness):
    """
    """
    F *= self.F_adp
    if F < 0.1:
      F = 0.1

    new_F = np.random.normal(F, 0.1)

    if new_F < 0.1:
      new_F = 0.1
    elif new_F > 2:
      new_F = 2

    if np.random.rand() < CR:
      return new_F
    else:
      return F

  def mutation(self):
    """
    V[i]=X[r1]+F(X[r2]-X[r3]),
    where i, r1, r2, r3 are randomly generated
    """
    X = self.X
    # i is not needed,
    # and TODO: r1, r2, r3 should not be equal
    random_idx = np.random.randint(0, self.size_pop, size=(self.size_pop, 3))

    r1, r2, r3 = random_idx[:, 0], random_idx[:, 1], random_idx[:, 2]

    if self.Y is not None:
      self.F = self.adaptive_random_F(self.F, self.CR, X, self.Y)
      print(self.F, self.Y.min())
    self.V = X[r1, :] + self.F * (X[r2, :] - X[r3, :])

    # the lower & upper bound still works in mutation
    mask = np.random.uniform(
        low=self.lb, high=self.ub, size=(self.size_pop, self.n_dim)
    )
    self.V = np.where(self.V < self.lb, mask, self.V)
    self.V = np.where(self.V > self.ub, mask, self.V)
    return self.V

  def run(self, max_iter=None):
    self.max_iter = max_iter or self.max_iter
    best = []
    for i in range(self.max_iter):
      self.mutation()
      self.crossover()
      self.U = roll(self.U, self.paning_step, self.n_len, self.panning_prob)
      self.selection()

      # record the best ones
      generation_best_index = self.Y.argmin()
      self.generation_best_X.append(self.X[generation_best_index, :].copy())
      self.generation_best_Y.append(self.Y[generation_best_index])
      self.all_history_Y.append(self.Y)
      if self.early_stop:
        best.append(min(self.generation_best_Y))
        if len(best) >= self.early_stop:
          if best.count(min(best)) == len(best):
            break
          else:
            best.pop(0)

    global_best_index = np.array(self.generation_best_Y).argmin()
    self.best_x = self.generation_best_X[global_best_index]
    self.best_y = self.func(np.array([self.best_x]))
    return self.best_x, self.best_y
