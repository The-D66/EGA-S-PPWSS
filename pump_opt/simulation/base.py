from __future__ import annotations
import logging

logging.getLogger()


class UndefinedError(Exception):
  pass


class BuildError(Exception):
  pass


class Float(float):
  version: int = 0

  def __add__(self, __value: float) -> Float:
    t = super().__add__(__value)
    t = Float(t)
    t.version = self.version
    return t

  def update(self, __value: float) -> Float:
    t = Float(__value)
    t.version = self.version
    return t


class Base:
  """
  init-> [run-> update-> check]
  """
  name: str = None
  input_flow: Float = Float(0)
  output_flow: Float = Float(0)
  input_waterlevel: Float = Float(0)
  output_waterlevel: Float = Float(0)
  power: float = 0.
  next_obj: Base = None
  before_obj: Base = None
  version: int = 0

  def print(self):
    log_line = "name: %20s, version:%4d, input_flow: %5.2f, output_flow: %5.2f, input_waterlevel: %5.2f, output_waterlevel: %5.2f, score: %5.4f" % (
        self.name, self.version, self.input_flow, self.output_flow,
        self.input_waterlevel, self.output_waterlevel, self.check()
    )
    print(log_line)
    logging.info(log_line)

  # def __init__(self) -> None:
  #   self.input_flow = None
  #   self.input_waterlevel = None

  def waterlevel_backpropagation(self, **kwargs):
    """
    """
    if self.before_obj:
      if self.input_waterlevel is not None and self.before_obj.output_waterlevel.version > self.input_waterlevel.version:
        self.input_waterlevel = self.before_obj.output_waterlevel
    if self.next_obj:
      if self.output_waterlevel is not None and self.next_obj.input_waterlevel.version > self.output_waterlevel.version:
        self.output_waterlevel = self.next_obj.input_waterlevel

  def flow_forward(self, *args, **kwargs):
    """
    """
    # if self.before_obj:
    #   if self.input_flow is not None and self.before_obj.output_flow.version < self.input_flow.version:
    #     self.before_obj.output_flow = self.input_flow
    #   if self.input_waterlevel is not None and self.before_obj.output_waterlevel.version < self.input_waterlevel.version:
    #     self.before_obj.output_waterlevel = self.input_waterlevel
    # if self.next_obj:
    #   if self.output_flow is not None and self.next_obj.input_flow.version < self.output_flow.version:
    #     self.next_obj.input_flow = self.output_flow
    #   if self.output_waterlevel is not None and self.next_obj.input_waterlevel.version < self.output_waterlevel.version:
    #     self.next_obj.input_waterlevel = self.output_waterlevel
    if self.before_obj:
      if self.input_flow is not None and self.before_obj.output_flow.version > self.input_flow.version:
        self.input_flow = self.before_obj.output_flow
    if self.next_obj:
      if self.output_flow is not None and self.next_obj.input_flow.version > self.output_flow.version:
        self.output_flow = self.next_obj.input_flow

  # def next(self):
  #   """"""
  #   raise NotDefineError

  def check(self):
    """
    """
    try:
      assert self.version == self.before_obj.version == self.next_obj.version
    except AttributeError:
      pass
    for var in self.__dict__:
      t = getattr(self, var)
      if isinstance(t, Float):
        assert t.version <= self.version
    return 0

  def init(self):
    """
    """
    def search_init(obj, var, iter_var):
      raw_name = obj.name
      while 1:
        if getattr(obj, var):
          logging.debug(f"INIT: {raw_name}.{var} from {obj.name}")
          return getattr(obj, var)
        elif getattr(obj, iter_var):
          obj = getattr(obj, iter_var)
        else:
          raise BuildError

    for var in self.__dict__:
      t = getattr(self, var)
      if isinstance(t, Float):
        t.version = 0
    # if self.next_obj:
    #   if self.next_obj.input_waterlevel is None and self.output_waterlevel is not None:
    #     self.next_obj.input_waterlevel = self.output_waterlevel
    #   if self.next_obj.input_flow is None and self.output_flow is not None:
    #     self.next_obj.input_flow = self.output_flow
    # if self.before_obj:
    #   if self.before_obj.output_waterlevel is None and self.input_waterlevel is not None:
    #     self.before_obj.output_waterlevel = self.input_waterlevel
    #   if self.before_obj.output_flow is None and self.input_flow is not None:
    #     self.before_obj.output_flow = self.input_flow

    if self.next_obj:
      if self.output_flow is None:
        self.output_flow = search_init(self.next_obj, "input_flow", "next_obj")
      if self.output_waterlevel is None:
        self.output_waterlevel = search_init(
            self.next_obj, "input_waterlevel", "next_obj"
        )
    if self.before_obj:
      if self.input_flow is None:
        self.input_flow = search_init(
            self.before_obj, "output_flow", "before_obj"
        )
      if self.input_waterlevel is None:
        self.input_waterlevel = search_init(
            self.before_obj, "output_waterlevel", "before_obj"
        )

  def batch_update(self, items: dict[str, Float]):
    for item, value in items.items():
      if value.version < self.version:
        logging.debug(
            f"UPDATE: {self.name}.{item}.version={value.version} +1 ->{self.version}"
        )
        value.version += 1
