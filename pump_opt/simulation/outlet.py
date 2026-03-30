from .base import Base, Float


class Outlet(Base):
  def __init__(
      self, name, input_flow, next_obj=None, before_obj=None, **kwargs
  ):
    self.name = name
    self.input_flow = Float(input_flow)
    self.next_obj = next_obj
    self.before_obj = before_obj

  def waterlevel_backpropagation(self, **kwargs):
    self.version += 1
    self.input_flow.version += 1

  def flow_forward(self, *args, **kwargs):
    if "input_flow" in kwargs:
      self.input_flow.real = kwargs["input_flow"]
    super().flow_forward(*args, **kwargs)
