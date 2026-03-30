import multiprocessing as mp
from copy import deepcopy

from pump_opt.simulation.model import Model


class Problem:
  """
  """
  def __init__(self, config_path="./data/model1.json") -> None:
    self.config_path = config_path
    self.model = Model(self.config_path)
    self.switch_time = self.model.config["switch_time"]

  def aim_func(self, pops):
    """
    """
    model = deepcopy(self.model)
    # if np.random.rand() < 0.005:
    #   score = modle.run(pops, 1)
    #   # print(score)
    #   print(int(np.log10(score)))
    # else:
    score = model.run(pops)
    return score

  def create_model(self, pops, **kwargs):
    """
    """
    model = deepcopy(self.model)
    model.run(pops, **kwargs)
    return model
