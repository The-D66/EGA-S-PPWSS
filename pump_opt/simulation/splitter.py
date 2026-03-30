from .base import Base, Float


class Splitter(Base):
  def __init__(self, name, next_obj: list[Base], before_obj: list[Base]):
    self.name = name
    self.next_obj = next_obj
    self.before_obj = before_obj
    self.waterlevel = None
    self.flow = None

  def init(self):
    for b_obj in self.before_obj:
      if b_obj.output_waterlevel:
        self.waterlevel = b_obj.output_waterlevel
    self.flow = Float(0)
    self.flow.version = 0
    for b_obj in self.before_obj:
      if b_obj.output_flow:
        self.flow += b_obj.output_flow

  def flow_forward(self):
    for n_obj in self.next_obj:
      pass

  @property
  def input_waterlevel(self):
    return self.waterlevel

  @input_waterlevel.setter
  def input_waterlevel(self, _):
    pass

  @property
  def output_waterlevel(self):
    return self.waterlevel

  @output_waterlevel.setter
  def output_waterlevel(self, _):
    pass

  @property
  def output_flow(self):
    return self.flow

  @output_flow.setter
  def output_flow(self, _):
    pass

  @property
  def input_flow(self):
    return self.flow

  @input_flow.setter
  def input_flow(self, _):
    pass
