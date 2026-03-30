import json

import numpy as np
import pandas as pd

from .base import Base, Float
# from .. import utils


class Tank(Base):
  def __init__(
      self,
      name,
      input_waterlevel,
      input_flow,
      bias_waterlevel=3,
      next_obj=None,
      before_obj=None,
      tank_path=None,
      output_flow=0,
      **kwargs
  ):
    self.name = name
    self.input_waterlevel = Float(input_waterlevel)
    self.init_waterlevel = Float(input_waterlevel)
    self.input_flow = Float(input_flow)
    self.next_obj: Base = next_obj
    self.before_obj: Base = before_obj
    self.bias_waterlevel = bias_waterlevel
    with open(tank_path, 'r', encoding='utf8') as f:
      self.config = json.load(f)
    self.volume = Float(self.waterlevel2volume(self.input_waterlevel))
    self.output_flow = Float(output_flow)
    if "delive" in kwargs:
      self.delive = kwargs["delive"]
    else:
      self.delive = 0
    if "output_waterlevel" in kwargs:
      self.output_waterlevel = Float(kwargs["output_waterlevel"])
    else:
      self.output_waterlevel = Float(input_waterlevel)
    if "outlet_output_flow" in kwargs:
      self.outlet_output_flow = kwargs["outlet_output_flow"]
    else:
      self.outlet_output_flow = 0
    self.kwargs = kwargs

  @property
  def delive_flow(self):
    if "sync_flow" in self.kwargs:
      sync_flow = self.kwargs["sync_flow"]
      if sync_flow:
        return self.input_flow - self.output_flow
    else:
      return self.outlet_output_flow

  def waterlevel2volume(self, waterlevel):
    """
    """
    def calc_volume(para, waterlevel):
      if len(para) == 3:
        return sum(
            np.array(para) * np.array([waterlevel * waterlevel, waterlevel, 1])
        )
      elif len(para) == 2:
        return sum(np.array(para) * np.array([waterlevel, 1]))
      elif len(para) == 1:
        return sum(np.array(para) * np.array([1]))
      elif len(para) == 0:
        return 0

    if isinstance(self.config["para"], list):
      return calc_volume(self.config["para"], waterlevel)
    elif isinstance(self.config["para"], dict):
      for key in self.config["para"]:
        if waterlevel < float(key):
          return calc_volume(self.config["para"][key]["para"], waterlevel)

  def volume2waterlevel(self, volume):
    """
    """
    para = self.config["para"]
    if isinstance(para, list):
      if len(para) == 3:
        if abs(para[0]) < 0.001:
          return (volume - para[2]) / para[1]
        else:
          if (para[1]**2 - 4 * para[0] * (para[2] - volume)) < 0:
            raise KeyError
          return (
              -para[1] + np.sqrt(para[1]**2 - 4 * para[0] * (para[2] - volume))
          ) / (2 * para[0])
      elif len(para) == 2:
        return (volume - para[1]) / para[0]
      elif len(para) == 1:
        return volume / para[0]
      else:
        return 0
    elif isinstance(para, dict):
      for key in para:
        if volume < float(para[key]["volume"]):
          if len(para[key]) == 3:
            if abs(para[key][0]) < 0.001:
              return (volume - para[key][2]) / para[key][1]
            else:
              if (
                  para[key][1]**2 - 4 * para[key][0] * (para[key][2] - volume)
              ) < 0:
                raise KeyError
              return (
                  -para[key][1] + np.sqrt(
                      para[key][1]**2 - 4 * para[key][0] *
                      (para[key][2] - volume)
                  )
              ) / (2 * para[key][0])
          elif len(para[key]) == 2:
            return (volume - para[key][1]) / para[key][0]
          elif len(para[key]) == 1:
            return volume / para[key][0]
          else:
            return 0

  def waterlevel_backpropagation(self, up_time, **kwargs):
    super().waterlevel_backpropagation(**kwargs)
    if "sync_flow" in self.kwargs:
      sync_flow = self.kwargs["sync_flow"]
      if sync_flow:
        self.input_waterlevel = self.input_waterlevel.update(
            self.output_waterlevel
        )
        if "adjust" in self.config:
          if self.input_waterlevel < self.config["adjust"]["min"]:
            self.input_waterlevel = self.input_waterlevel.update(
                self.config["adjust"]["min"]
            )

        self.batch_update({"input_waterlevel": self.input_waterlevel})
        return
    else:
      self.volume += (
          self.input_flow - self.output_flow - self.outlet_output_flow /
          (1 - self.delive)
      ) * up_time / 1e4
      waterlevel = self.volume2waterlevel(self.volume)
      if "adjust" in self.config:
        if waterlevel < self.config["adjust"]["min"]:
          waterlevel = self.config["adjust"]["min"]
      self.input_waterlevel = self.input_waterlevel.update(waterlevel)
      self.output_waterlevel = self.output_waterlevel.update(waterlevel)
      self.batch_update(
          {
              "volume": self.volume,
              "input_waterlevel": self.input_waterlevel,
              "output_waterlevel": self.output_waterlevel
          }
      )

    next_obj = self.next_obj
    obj = self
    while next_obj is not None:
      if next_obj.input_waterlevel.version < obj.output_waterlevel.version:
        waterlevel = self.volume2waterlevel(self.volume)
        next_obj.input_waterlevel = obj.output_waterlevel.update(waterlevel)
      else:
        break
      obj = next_obj
      next_obj = obj.next_obj

  def check(self):
    super().check()
    if self.volume < 0:
      return 1
    score = 0
    if "adjust" in self.config:
      waterlevel = self.volume2waterlevel(self.volume)
    else:
      waterlevel = self.input_waterlevel
    if waterlevel > max(self.config["range"]):
      score += 0.1 + 0.8 * min(
          1,
          abs(waterlevel - max(self.config["range"])) /
          max(self.config["range"])
      )
    elif waterlevel < min(self.config["range"]):
      score += 0.1 + 0.8 * min(
          1,
          abs(waterlevel - min(self.config["range"])) /
          min(self.config["range"])
      )
    if abs(waterlevel - self.init_waterlevel) > self.bias_waterlevel:
      score += 1
    return score

  def flow_forward(self, *args, **kwargs):
    self.version += 1
    super().flow_forward(*args, **kwargs)
    if "sync_flow" in self.kwargs:
      sync_flow = self.kwargs["sync_flow"]
      if sync_flow:
        self.output_flow = self.output_flow.update(
            self.input_flow * (1 - self.delive)
        )
        self.batch_update({"output_flow": self.output_flow})
        return
