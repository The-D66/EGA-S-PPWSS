import json
import logging
from dataclasses import dataclass
from typing import List, Union

import numpy as np


@dataclass
class Unit:
  name: str = "None"
  input_waterlevel: float = 0
  output_waterlevel: float = 0
  input_flow: float = 0
  output_flow: float = 0


@dataclass
class API:
  """
  """
  area: str = "sC-sA"
  switch_max: float = 5
  aim_vol: float = 1500000
  method: str = "eco"
  allow_period: Union[str, List[int], None] = None
  setting_unit: Union[List[Unit], None] = None
  eff_flag: bool = True
  switch_flag: bool = False
  flow_flag: bool = True
  plot_flag: bool = False
  waterlevel_flag: bool = True
  output: str = "./result/resusB.json"
  start_time: int = 8
  json_input: Union[str,
                    None] = None
  log: str = "INFO"

  @property
  def log_level(self):
    if self.log == "INFO":
      return logging.INFO
    elif self.log == "DEBUG":
      return logging.DEBUG
    elif self.log == "WARNING":
      return logging.WARNING
    elif self.log == "ERROR":
      return logging.ERROR
    elif self.log == "CRITICAL":
      return logging.CRITICAL
    else:
      raise ValueError("log level error")

  @staticmethod
  def update_para(para, unit, data, name):
    """
    """
    if unit[para] > -0.5 and name in data["unit"] and para in data["unit_para"][
        name]:
      data["unit_para"][name][para] = unit[para]

  @property
  def total_time(self):
    """
    """
    if self.start_time < 8:
      return 8 - self.start_time
    else:
      return 32 - self.start_time

  @property
  def start_time_idx(self):
    """
    """
    if self.start_time < 8:
      return 16 + self.start_time
    else:
      return self.start_time - 8

  def to_json(self):
    """

    Returns:
    """
    if isinstance(self.allow_period, str):
      self.allow_period = np.fromstring(
          self.allow_period[1:-1], dtype=int, sep=','
      ).tolist()
    if self.json_input is not None:
      try:
        json_input = json.loads(self.json_input)
        super().__init__(**json_input)
      except json.JSONDecodeError:
        pass
    with open("./data/area/%s.json" % self.area, 'r', encoding='utf8') as f:
      data = json.load(f)
    data["switch_max"] = self.switch_max
    data["aim_vol"] = self.aim_vol
    data["method"] = self.method
    data["total_time"] = self.total_time
    data["start_time_idx"] = self.start_time_idx
    if self.allow_period is not None:
      data["allow_period"] = self.allow_period
    elif "allow_period" not in data:
      data["allow_period"] = [1] * 24
    if self.setting_unit is not None and self.setting_unit != "":
      self.setting_unit = json.loads(self.setting_unit)
      for unit in self.setting_unit:
        for para in self.setting_unit[unit]:
          self.update_para(para, self.setting_unit[unit], data, unit)
    data = json.dumps(data, ensure_ascii=False)
    return data

  @property
  def lb(self):
    """
    
    """
    total_time = self.total_time
    if "sC-sA" in self.area:
      return [0] * total_time
    elif "sC-sB" in self.area:
      return [0] * total_time * 2
    elif "sC-sD" in self.area:
      return [0] * total_time + [12] * total_time + [0] * total_time
    elif "sA-sB" in self.area:
      return [0] * total_time
    elif "sA-sD" in self.area:
      return [0] * total_time * 2
    elif "sB-sD" in self.area:
      return [0] * total_time
    else:
      raise ValueError("area error")

  @property
  def ub(self):
    total_time = self.total_time
    if "sC-sA" in self.area:
      return [85] * total_time
    elif "sC-sB" in self.area:
      return [85] * total_time + [65] * total_time
    elif "sC-sD" in self.area:
      return [85] * total_time + [65] * total_time + [35] * total_time
    elif "sA-sB" in self.area:
      return [65] * total_time
    elif "sA-sD" in self.area:
      return [65] * total_time + [35] * total_time
    elif "sB-sD" in self.area:
      return [35] * total_time
    else:
      raise ValueError("area error")

  @property
  def s_time(self):
    return len(self.lb)

  @property
  def n_len(self):
    n_len = [self.total_time]
    if "sC-sA" in self.area:
      return n_len
    elif "sC-sB" in self.area:
      return n_len * 2
    elif "sC-sD" in self.area:
      return n_len * 3
    elif "sA-sB" in self.area:
      return n_len
    elif "sA-sD" in self.area:
      return n_len * 2
    elif "sB-sD" in self.area:
      return n_len
    else:
      raise ValueError("area error")


if __name__ == '__main__':
  api = API(
      area="sC-sD", setting_unit=[{
          "name": "sE_tank",
          "input_flow": 1e10
      }]
  )
  print(api.to_json())
