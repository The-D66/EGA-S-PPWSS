"""
"""
import json
import sys
import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from .base import Base, Float

logging.getLogger()

sys.path.append("..")
# from .. import utils

G = 9.81
Rho = 1e3


class Pump(Base):
  """
  """
  def __init__(
      self,
      name,
      input_waterlevel,
      output_waterlevel,
      input_flow,
      # control,
      pump_path=None,
      next_obj: Base = None,
      before_obj: Base = None,
      start_time_idx: int = 0,
      **kwargs
  ):
    self.name = name
    self.input_waterlevel = Float(input_waterlevel)
    self.output_waterlevel = Float(output_waterlevel)
    self.input_flow = Float(input_flow)
    # self.control = control
    with open(pump_path, 'r', encoding='utf8') as f:
      self.config = json.load(f)
    eff_table = pd.read_csv(self.config["eff"], delimiter=r'\s+', comment='%')
    freq_table = eff_table.filter(regex=r"n\d+")
    freq_table = freq_table[freq_table > 0]
    freq_table = freq_table[freq_table > self.config["base_freq"] + 1]
    freq_table = freq_table[freq_table < self.config["base_freq"] - 1]
    eff_table["base_freq_num"] = freq_table.notna().sum(axis=1)
    self.eff_table = eff_table[["Q", "H", "num", "η", "base_freq_num"]]
    self.eff_table["η"] /= 100
    if "lift" in self.config:
      self.eff_table = self.eff_table[
          (self.eff_table['H'] >= self.config["lift"][0]) &
          (self.eff_table['H'] <= self.config["lift"][1])]
    if "flow" in self.config:
      self.eff_table = self.eff_table[
          (self.eff_table['Q'] >= self.config["flow"][0]) &
          (self.eff_table['Q'] <= self.config["flow"][1])]
    self.unique_flow = pd.unique(self.eff_table['Q'])
    self.unique_lift = pd.unique(self.eff_table['H'])
    self.unique_num = pd.unique(self.eff_table["num"])
    self.eff_table = self.eff_table.values[:, 2:]
    self.output_flow = Float(input_flow)
    self.opening_num = 0
    self.past_opening_num = 0
    self.freq_num = 0
    self.next_obj = next_obj
    self.before_obj = before_obj
    self.kwargs = kwargs
    if "aim_vol" in kwargs:
      self.aim_vol_rate = float(self.kwargs["aim_vol"])
    else:
      self.aim_vol_rate = 1.
    if "switch_threshold" in kwargs:
      self.switch_threshold = float(self.kwargs["switch_threshold"])
    else:
      self.switch_threshold = 0
    if "freq_threshold" in kwargs:
      self.freq_threshold = float(self.kwargs["freq_threshold"])
    else:
      self.freq_threshold = 0

  @staticmethod
  def find_nearest(array, value):
    return (np.abs(np.asarray(array) - value)).argmin()

  def search_eff(
      self, flow, lift, output_num=True, past_num=0, output_freq_num=True
  ):
    flow_idx = self.find_nearest(self.unique_flow, flow)
    lift_idx = self.find_nearest(self.unique_lift, lift)
    idx = len(self.unique_num) * (len(self.unique_lift) * flow_idx + lift_idx)
    eff = 0
    same_eff = 0
    num = 100
    logging.debug("PUMP.SEARCH_EFF: %s, old_pump_num:%s", self.name, past_num)
    for i in self.unique_num:
      eff_data = self.eff_table[idx + i - 1]
      logging.debug(
          "PUMP.SEARCH_EFF: %s, flow:%s, lift:%s, pump_num:%s, eff:%s",
          self.name, flow, lift, i, eff_data[-1]
      )
      # assert abs(eff_data["Q"] - flow) < 1 and abs(eff_data["H"] - lift) < 1
      if past_num and eff_data[0] == past_num:
        same_eff = eff_data[1]
      if eff_data[1] > eff:
        eff = eff_data[1]
        num = eff_data[0]
    if self.switch_threshold and (
        eff - same_eff
    ) < self.switch_threshold + 1e-4:
      eff = same_eff
      num = past_num
    if num < 50 and output_freq_num:
      self.freq_num = num - self.eff_table[int(idx + num - 1)][2]
    if output_num:
      return eff, num
    return eff

  def waterlevel_backpropagation(self, **kwargs):
    super().waterlevel_backpropagation(**kwargs)
    self.past_opening_num = self.opening_num
    _, self.opening_num = self.search_eff(
        self.input_flow, self.lift, past_num=self.past_opening_num
    )

  def flow_forward(self, *args, **kwargs):
    self.version += 1
    super().flow_forward(*args, **kwargs)
    if "input_flow" in kwargs:
      if kwargs["input_flow"] > max(self.unique_flow):
        input_flow = max(self.unique_flow)
      elif kwargs["input_flow"] < 4:
        input_flow = 0
      elif kwargs["input_flow"] < min(self.unique_flow):
        input_flow = min(self.unique_flow)
      else:
        input_flow = kwargs["input_flow"]
      self.input_flow = self.input_flow.update(input_flow)
      self.output_flow = self.output_flow.update(input_flow)

    if self.output_flow < 5:
      self.output_flow = self.output_flow.update(0)
      self.input_flow = self.input_flow.update(0)

    self.batch_update(
        {
            "input_flow": self.input_flow,
            "output_flow": self.output_flow
        }
    )

    before_obj = self.before_obj
    obj = self
    while before_obj is not None:
      if before_obj.output_flow.version < obj.input_flow.version:
        before_obj.output_flow = obj.input_flow
      else:
        break
      obj = before_obj
      before_obj = obj.before_obj

  @property
  def lift(self):
    return self.output_waterlevel - self.input_waterlevel

  @property
  def power(self):
    if self.input_flow < 5:
      return 0
    return self.lift * self.input_flow * G * Rho / (
        self.eff + 1e-8
    ) / 3600 / 1000

  @property
  def eff(self):
    if self.output_flow < 5:
      return 0
    return self.search_eff(
        self.input_flow,
        self.lift,
        past_num=self.past_opening_num,
        output_num=False
    )

  def check(self):
    super().check()

    score = 0

    if self.input_waterlevel > max(self.config["input_waterlevel"]):
      score += 0.5 * min(
          1,
          abs(self.input_waterlevel - max(self.config["input_waterlevel"])) /
          max(self.config["input_waterlevel"])
      )
    elif self.input_waterlevel < min(self.config["input_waterlevel"]):
      score += 0.5 * min(
          1,
          abs(self.input_waterlevel - min(self.config["input_waterlevel"])) /
          min(self.config["input_waterlevel"])
      )

    if self.output_waterlevel > max(self.config["output_waterlevel"]):
      score += 0.5 * min(
          1,
          abs(self.output_waterlevel - max(self.config["output_waterlevel"])) /
          max(self.config["output_waterlevel"])
      )
    elif self.output_waterlevel < min(self.config["output_waterlevel"]):
      score += 0.5 * min(
          1,
          abs(self.output_waterlevel - min(self.config["output_waterlevel"])) /
          min(self.config["output_waterlevel"])
      )
    return score

  def print(self):
    log_line = "name: %20s, version:%4d, input_flow: %5.2f, power: %11.2f, input_waterlevel: %5.2f, output_waterlevel: %5.2f, score: %5.4f" % (
        self.name, self.version, self.input_flow, self.power,
        self.input_waterlevel, self.output_waterlevel, self.check()
    )
    print(log_line)
    logging.info(log_line)
    if self.power > 1000:
      pass

  def final_check(self):
    """
    TODO
    """
    score = self.check()

    # def output_eff(self):
    #   eff, _ = self.search_eff(
    #       self.input_flow, self.output_waterlevel - self.input_waterlevel
    #   )
    #   unit_energy = 2.72 / eff

  def make_init_flow(
      self,
      aim_vol=None,
      switch_time=60,
      mode="eco",
      allow_period=None,
      time_bias=0,
      **kwargs
  ):
    aim_vol *= self.aim_vol_rate
    avl_flow = {}
    bill = np.array(self.config["bill"])[time_bias:]
    bill = np.repeat(bill, int((86400 / switch_time / 60) / 24))
    if mode == "eco":
      sorted_bill = np.argsort(-bill[time_bias:], kind="stable")[::-1]
    elif mode == "eff":
      sorted_bill = np.arange(len(bill[time_bias:]) - 1, -1, -1, dtype=np.int_)
    elif mode == "fast":
      sorted_bill = np.arange(0, len(bill[time_bias:]), 1, dtype=np.int_)
    elif mode == "bwq":
      if allow_period is None:
        raise ValueError("allow_period must be given when mode is 'bwq'")
      else:
        allow_period = np.array(allow_period)
        sorted_bill = np.argsort(-allow_period[time_bias:], kind="stable")
    else:
      raise ValueError("mode must be one of 'eco', 'eff', 'fast', 'bwq'")
    for per in range(0, 101, 5):
      flow = max(self.unique_flow) * (per / 100)
      if flow < 0.5:
        continue
      avg_time = aim_vol / flow / switch_time / 60
      if avg_time <= 86400 / switch_time / 60 - time_bias:
        flow_list = np.zeros_like(bill).astype(float)
        flow_list[sorted_bill[:min(
            int(86400 / switch_time / 60 - time_bias), int(np.ceil(avg_time))
        )]] = flow
        avl_flow[per] = flow_list
        logging.debug(
            "PUMP.MAKE_INIT_FLOW: %s, method:%s, per:%2d, flow:%s", self.name,
            mode, per, str(flow_list.tolist())
        )
    return avl_flow
