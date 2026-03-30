import json
import time
from typing import Dict, Union

import numpy as np

from .pipe import Pipe
from .pump import Pump
from .tank import Tank
from .base import Base


class Model:
  """
  """
  fund = 0

  def __init__(self, config_path="./data/model1.json"):
    try:
      self.base_config = json.loads(config_path)
    except json.JSONDecodeError:
      with open(config_path, 'r', encoding='utf8') as f:
        self.base_config = json.load(f)
    self.name = self.base_config["name"]
    self.aim_vol = self.base_config["aim_vol"]
    self.switch = 0
    self.switch_minutes = self.base_config["switch_time"]
    self.total_time = self.base_config["total_time"]
    # self.total_time = 24 * 60 // self.switch_minutes
    self.start_time_idx = self.base_config["start_time_idx"]
    self.freq_time = 0
    self.freq_max = self.base_config["freq_max"]
    self.up_time = self.switch_minutes * 60
    self.score = 0
    self.unit: Dict[str, Base] = {}
    self.weights: Dict[str, float] = {}
    self.control_item = {}
    self.log = {}
    self.bill = None
    for name in self.base_config["unit"]:
      self.unit[name] = self.build_unit(name)
      if "control" in self.base_config["unit_para"][name]:
        self.control_item[name] = self.base_config["unit_para"][name]["control"]
        if self.bill is None and self.unit[name].config["bill"]:
          self.bill = self.unit[name].config["bill"]
      self.weights[name] = self.base_config["unit_para"][name]["weights"]
    self.switch_max = self.base_config["switch_max"] * len(self.control_item)
    if "method" in self.base_config:
      self.method = self.base_config["method"]
    else:
      self.method = "eco"
    self.init()

  @property
  def config(self) -> Dict:
    config = self.base_config.copy()
    for key in config:
      if hasattr(self, key):
        config[key] = getattr(self, key)
    return config

  def build(self) -> Dict[str, Union[Base, Tank, Pump, Pipe]]:
    for name in self.base_config["unit"]:
      self.unit[name] = self.build_unit(name)
      self.weights[name] = self.base_config["unit_para"][name]["weights"]
    self.init()
    return self.unit

  def build_unit(self, name) -> Base:
    config = self.base_config["unit_para"][name]
    if config["type"] == "pump":
      return Pump(**config)
    elif config["type"] == "pipe":
      return Pipe(**config)
    elif config["type"] == "tank":
      return Tank(**config)
    else:
      raise TypeError

  def init(self):
    self.switch = 0
    for unit in self.unit.values():
      if unit.before_obj and isinstance(unit.before_obj, str):
        unit.before_obj = self.unit[unit.before_obj]
        unit.before_obj.next_obj = unit
      if unit.next_obj and isinstance(unit.next_obj, str):
        unit.next_obj = self.unit[unit.next_obj]
        unit.next_obj.before_obj = unit

    for unit in self.unit.values():
      unit.init()

    for unit in self.unit.values():
      unit.check()

    if self.method == "eco":
      self.run_step = self.eco_run_step
    elif self.method == "eff":
      self.run_step = self.eff_run_step
    elif self.method == "fast":
      self.run_step = self.fast_run_step
    elif self.method == "bwq":
      self.run_step = self.bwq_run_step
    else:
      raise ValueError("method error")

  def simulate(self, idx=0, control=None, **kwargs):
    if control is None:
      control = {}
    for unit in list(self.unit.values()):
      if unit.name in control:
        unit.flow_forward(**control[unit.name])
      else:
        unit.flow_forward()
    for unit in list(self.unit.values())[::-1]:
      unit.waterlevel_backpropagation(up_time=self.up_time)

  def eco_run_step(self, idx=0, control=None, **kwargs):
    """ 
    """
    self.simulate(idx, control, **kwargs)

    score = 0
    fund = 0
    for name, unit in self.unit.items():
      score += self.weights[name] * unit.check()
      if type(unit) == Pump:
        if unit.opening_num != unit.past_opening_num:
          self.switch += 1
        pump_fund = unit.power * unit.config["bill"][
            self.start_time_idx + idx // int(60 // self.switch_minutes)] / 100
        score += pump_fund
        fund += pump_fund
      else:
        score += unit.power
    # if self.switch > self.switch_max:
    #   score += 1e4 * self.switch / self.switch_max
    return score, fund

  def fast_run_step(self, idx=0, control=None, **kwargs):
    """ 
    """
    self.simulate(idx, control, **kwargs)

    score = 0
    for name, unit in self.unit.items():
      score += self.weights[name] * unit.check()
      if type(unit) == Pump:
        if unit.opening_num != unit.past_opening_num:
          self.switch += 1
        score += unit.input_flow * (idx + 1)
      # else:
      #   score += unit.power
    return score, 0.

  def eff_run_step(self, idx=0, control=None, **kwargs):
    """ 
    """
    self.simulate(idx, control, **kwargs)

    score = 0
    fund = 0
    for name, unit in self.unit.items():
      score += self.weights[name] * unit.check()
      if type(unit) == Pump:
        if unit.opening_num != unit.past_opening_num:
          self.switch += 1
        score += unit.power
        fund += unit.power
      else:
        score += unit.power
    return score, fund

  def bwq_run_step(self, idx=0, control=None, **kwargs):
    """ 
    """
    if control is None:
      control = {}
    for unit in list(self.unit.values()):
      if isinstance(unit, Pump):
        if unit.name in control:
          control[unit.name]["input_flow"] = control[
              unit.name]["input_flow"] * self.config["allow_period"][
                  self.start_time_idx + idx]
          break

    self.simulate(idx, control, **kwargs)

    score = 0
    for name, unit in self.unit.items():
      score += self.weights[name] * unit.check()
      if type(unit) == Pump:
        if unit.opening_num != unit.past_opening_num:
          self.switch += 1
        score += unit.power * unit.config["bill"][
            self.start_time_idx + idx // int(60 // self.switch_minutes)] / 100
      else:
        score += unit.power
    return score, 0.

  @staticmethod
  def trans_result(pop):
    # pop = pop[:len(pop) // 2]
    pop = np.array(pop)
    pop[pop < 3] = 0
    return pop

  def trans_result2(self, pop):
    flow_list = np.zeros(len(pop) // 2)
    f = pop[0] if pop[0] >= 5 else 0
    for i in range(self.total_time):
      # rate = pop[i + self.s_time]
      # if rate > .8:
      #   rate = 1
      # elif rate < .2:
      #   rate = 0
      # else:
      #   rate = (rate - .2) / .6

      # if f > 0:
      #   f = pop[i] * rate + f * (1 - rate)
      #   f = f if f >= 5 else 0
      # else:
      #   f = pop[i] if pop[i] >= 5 else 0
      # flow_list[i] = f
      if pop[self.total_time + i] > 0.5:
        f = pop[i] if pop[i] >= 5 else 0
        flow_list[i] = f
      else:
        flow_list[i] = f
    return flow_list

  def run(
      self,
      pops,
      print_flag=False,
      log_eff_flag=False,
      log_switch_flag=False,
      log_flow_flag=False,
      log_waterlevel_flag=False,
      log_opening_num_flag=True,
      **kwargs
  ):
    """
    """

    # a = time.time()
    # print(0, a - a)

    # print(1, time.time() - a)
    flow_list = self.trans_result(pops)
    # self.build()
    score = 0
    fund = 0
    if self.method == "bwq":
      volume = sum(
          flow_list[:self.total_time] *
          np.array(self.config["allow_period"][self.start_time_idx:])
      ) * self.switch_minutes * 60
    else:
      volume = sum(flow_list[:self.total_time]) * self.switch_minutes * 60
    # print(volume, abs(volume - 2000000))
    # print(2, time.time() - a)
    vol_rate = volume / self.aim_vol
    if vol_rate > 1.05:
      pass
    elif vol_rate < 0.95:
      score += 1e5 + 1e4 * abs(volume - self.aim_vol) / self.aim_vol
    elif vol_rate < 0.85:
      score += 1e8 + 1e7 * abs(volume - self.aim_vol) / self.aim_vol
    elif vol_rate < 0.75:
      score += 1e10 + 1e9 * abs(volume - self.aim_vol) / self.aim_vol
    elif vol_rate < 0.6:
      score += 1e12 + 1e1 * abs(volume - self.aim_vol) / self.aim_vol
      # return score

    # elif volume > self.aim_vol * 1.05:
    #   score += 1e3 + 1e3 * abs(volume - self.aim_vol) / self.aim_vol
    # print(3, time.time() - a)
    for idx in range(self.total_time):
      # if pops[self.s_time + idx] > 0.5:
      #   c = pops[idx]
      #   self.freq_time += 1
      control = {}
      for unit, priority in self.control_item.items():
        control[unit] = {
            "input_flow": flow_list[idx + (priority - 1) * self.total_time]
        }
      # control = {"sA_pump": {"input_flow": flow_list[idx]}}

      score_one_time = self.run_step(idx, control, **kwargs)
      score += score_one_time[0]
      fund += score_one_time[1]

      if print_flag:
        self.print()
        print(
            "score: %10.2f, bill: %4.0f, switch time: %2d, switch max: %2d" %
            (score, self.bill[idx], self.switch, self.switch_max)
        )
      if log_eff_flag:
        self.output_pump_cost(idx)
      if log_opening_num_flag:
        self.output_pump_opening_num(idx)
      if log_switch_flag:
        self.output_switch(idx)
      if log_flow_flag:
        for unit in self.unit.values():
          if isinstance(unit, Pump):
            if "%s_flow" % unit.name in self.log:
              self.log["%s_flow" % unit.name][idx] = unit.input_flow
            else:
              self.log["%s_flow" % unit.name] = [0] * self.total_time
              self.log["%s_flow" % unit.name][idx] = unit.input_flow
            # if idx in self.log:
            #   if "pump_flow" in self.log[idx]:
            #     self.log[idx]["pump_flow"][idx] = unit.input_flow
            #   self.log[idx]["%s_flow" % unit.name][idx] = unit.input_flow
            # else:
            #   self.log[idx] = {"%s_flow" % unit.name: [0] * self.s_time}
            #   self.log[idx]["%s_flow" % unit.name][idx] = unit.input_flow
      if log_waterlevel_flag:
        self.output_waterlevel(idx)

    #   score += 5e3 * self.freq_time / self.freq_max
    # print(4, time.time() - a)
    if self.switch_max:
      if self.switch > self.switch_max:
        score *= 2**(self.switch - self.switch_max)
      else:
        score *= 0.997**(self.switch_max - self.switch)
    self.score = score
    if self.method in ("eco", "eff"):
      self.fund = fund
    elif self.method == "fast":
      self.fund = self._find_last_non_zero_index(flow_list[:self.total_time])
    elif self.method == "bwq":
      self.fund = vol_rate * 100
    return score

  @staticmethod
  def _find_last_non_zero_index(flow_list):
    for index, value in enumerate(reversed(flow_list)):
      if value != 0:
        return len(flow_list) - index - 1
    return None

  def check(self):
    score = 0
    for unit in self.unit.values():
      score += unit.check()
    return score

  def print(self):
    print()
    print(time.asctime())
    for unit in self.unit.values():
      unit.print()
    # print(self.switch)

  def output_waterlevel(self, idx):
    waterlevel_dict = {}
    for unit in self.unit.values():
      waterlevel_dict[unit.name] = {}
      if isinstance(unit, Tank):
        waterlevel_dict[unit.name]["input"] = unit.volume2waterlevel(
            unit.volume
        )
        waterlevel_dict[unit.name]["output"] = unit.volume2waterlevel(
            unit.volume
        )
      else:
        waterlevel_dict[unit.name]["input"] = unit.input_waterlevel
        waterlevel_dict[unit.name]["output"] = unit.output_waterlevel

    if idx in self.log:
      self.log[idx]["waterlevel"] = waterlevel_dict
    else:
      self.log[idx] = {"waterlevel": waterlevel_dict}

  def output_pump_opening_num(self, idx):
    opening_num_dict = {}
    freq_num_dict = {}
    for unit in self.unit.values():
      if isinstance(unit, Pump):
        if unit.power < 0.1:
          opening_num_dict[unit.name] = 0
          freq_num_dict[unit.name] = 0
        else:
          opening_num_dict[unit.name] = unit.opening_num
          freq_num_dict[unit.name] = unit.freq_num
    if idx in self.log:
      self.log[idx]["opening_num"] = opening_num_dict
    else:
      self.log[idx] = {"opening_num": opening_num_dict}
    self.log[idx]["freq_num"] = freq_num_dict

  def output_pump_cost(self, idx):
    unit_energy_dict = {}
    unit_cost_dict = {}
    power_dict = {}
    for unit in self.unit.values():
      if isinstance(unit, Pump):
        eff = unit.eff
        power = unit.power
        if eff > .2:
          unit_energy = 2.72 / eff
        else:
          unit_energy = 0
        unit_energy_dict[unit.name] = unit_energy
        unit_cost_dict[unit.name] = unit_energy * unit.config["bill"][
            idx // int(60 // self.switch_minutes)] / 100
        power_dict[unit.name] = power

    if idx in self.log:
      self.log[idx]["unit_energy"] = unit_energy_dict
      self.log[idx]["unit_cost"] = unit_cost_dict
      self.log[idx]["power"] = power_dict
    else:
      self.log[idx] = {"unit_energy": unit_energy_dict}
      self.log[idx]["unit_cost"] = unit_cost_dict
      self.log[idx]["power"] = power_dict

  def output_switch(self, idx):
    if idx in self.log:
      self.log[idx]["switch"] = self.switch
    else:
      self.log[idx] = {"switch": self.switch}

  def make_init_pop(self):
    """
    """
    pop_dict = {}
    pop_count = 0
    for unit in self.unit.values():
      if isinstance(unit, Pump):
        init_flow = unit.make_init_flow(
            mode=self.method, time_bias=self.start_time_idx, **self.config
        )
        pop_dict[unit.name] = init_flow
        pop_count += self.total_time

    init_flow_list = []
    for percent in range(0, 101, 5):
      init_flow = []
      for _, pop in pop_dict.items():
        if percent not in pop:
          break
        init_flow += pop[percent].tolist()
      if len(init_flow) < pop_count:
        continue
      init_flow_list.append(init_flow)
    return np.array(init_flow_list)

  def warmup(self):
    """
    TODO
    """
    for unit in self.unit.values():
      unit.warmup()
