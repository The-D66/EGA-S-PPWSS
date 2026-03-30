import json

import numpy as np
import pandas as pd

from .base import Base, Float
from .base import UndefinedError


class Pipe(Base):
  def __init__(
      self,
      name,
      input_flow,
      input_waterlevel,
      config_path,
      output_flow=0,
      output_waterlevel=0,
      next_obj=None,
      before_obj=None,
      compute_obj="flow",
      **kwargs
  ):
    self.name = name
    self.input_flow = Float(input_flow)
    self.output_flow = Float(input_flow)
    self.input_waterlevel = Float(input_waterlevel)
    self.output_waterlevel = Float(output_waterlevel)
    self.next_obj = next_obj
    self.before_obj = before_obj
    with open(config_path, 'r', encoding='utf8') as f:
      self.config = json.load(f)
    self.compute_obj = compute_obj

  def flow2loss(self, flow):
    """
    """
    return sum(np.array(self.config["para"]) * np.array([flow * flow, flow, 1]))

  def loss2flow(self, loss):
    """
    """
    para = self.config["para"]
    if (para[1]**2 - 4 * para[0] * (para[2] - loss)) < 0:
      return 0
    return (-para[1] + np.sqrt(para[1]**2 - 4 * para[0] *
                               (para[2] - loss))) / (2 * para[0])

  def waterlevel_backpropagation(self, **kwargs):
    super().waterlevel_backpropagation(**kwargs)

    # self.before_obj.output_flow = self.input_flow
    if self.compute_obj == "waterlevel":
      # self.output_flow = Float(self.input_flow)
      self.input_waterlevel = self.input_waterlevel.update(
          self.output_waterlevel + self.flow2loss(self.output_flow)
      )
      self.batch_update({"input_waterlevel": self.input_waterlevel})
      # self.before_obj.output_waterlevel = self.input_waterlevel
      # self.before_obj.output_flow = self.input_flow = self.output_flow
    else:
      raise UndefinedError

  def flow_forward(self, *args, **kwargs):
    self.version += 1
    super().flow_forward(*args, **kwargs)
    if self.compute_obj == "flow":
      flow = self.loss2flow(abs(self.output_waterlevel - self.input_waterlevel))
      self.input_flow = self.input_flow.update(flow)
      if "loss" in self.config:
        loss = self.config["loss"]
      else:
        loss = 0
      self.output_flow = self.output_flow.update(flow * (1 - loss))
      self.batch_update(
          {
              "input_flow": self.input_flow,
              "output_flow": self.output_flow
          }
      )
    elif self.compute_obj == "waterlevel":
      self.output_flow = self.output_flow.update(self.input_flow)
      self.batch_update({"output_flow": self.output_flow})
    else:
      raise UndefinedError
