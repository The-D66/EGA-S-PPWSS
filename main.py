import os
import json
import logging
import datetime

import sko
import simple_parsing
import numpy as np

from pump_opt.problem import Problem
from pump_opt.optimization.panning_GA import Panning_GA
from pump_opt.api import API

log_dir = "logs"
if not os.path.exists(log_dir):
  os.makedirs(log_dir)

current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = os.path.join(log_dir, f"{current_time}.log")

if os.name == "nt":
  fmt = '%(asctime)s|%(levelname)-8s|%(thread)-9d|%(message)s'
else:
  fmt = '%(asctime)s|%(levelname)-8s|%(process)-9d|%(message)s'

with open("./data/config.json", 'r', encoding="utf8") as f:
  config = json.load(f)

parser = simple_parsing.ArgumentParser()
parser.add_argument("--setting_unit", type=str, default="", help="foo help")
# parser.add_argument("--allow_period", type=str, default="", help="foo help")

parser.add_arguments(API, dest="options")
args = parser.parse_args()

options: API = args.options
options.setting_unit = args.setting_unit

logging.basicConfig(
    level=options.log_level,
    format=fmt,
    datefmt='%Y/%m/%d %H:%M:%S',
    filename=log_filename,
    filemode='w'
)
# options.allow_period = args.allow_period

EFF_FLAG = options.eff_flag
SWITCH_FLAG = options.switch_flag
FLOW_FLAG = options.flow_flag
PLOT_FLAG = options.plot_flag

PRINT_FLAG = True

output_file = options.output

options_json = options.to_json()
logging.debug(options_json)

pro = Problem(options_json)
s_time = 24 * 60 // pro.switch_time
result = {}

sko.tools.set_run_mode(pro.aim_func, "multiprocessing")

ga = Panning_GA(
    pro.aim_func,
    options.s_time,
    lb=options.lb,
    ub=options.ub,
    size_pop=config["size_pop"],
    max_iter=config["max_iter"],
    prob_mut=config["prob_mut"],
    precision=config["precision"],
    early_stop=config["early_stop"],
    n_len=options.n_len
)
init_flow = pro.model.make_init_pop()
logging.debug(str(init_flow))
ga.Chrom = ga.x2chrom(init_flow)
pop, fund = ga.run()
# pop = init_flow[-1]

result["pop"] = list(pop)
mo = pro.create_model(
    pop,
    log_eff_flag=EFF_FLAG,
    log_switch_flag=SWITCH_FLAG,
    log_flow_flag=FLOW_FLAG,
    print_flag=PRINT_FLAG,
    log_waterlevel_flag=options.waterlevel_flag
)
result["flow"] = list(mo.trans_result(pop))
if mo.method in ("eco", "eff"):
  result["fund"] = float(mo.fund) * .36 if mo.fund < 1e6 else 0
else:
  result["fund"] = float(mo.fund)
if EFF_FLAG or SWITCH_FLAG or FLOW_FLAG:
  result["pump_eff"] = mo.log
print(mo.score, mo.fund)
logging.info(
    "method:%s, score: %.2f, fund: %.2f", mo.method, mo.score, result['fund']
)
logging.debug(result["pump_eff"])

with open(output_file, 'w', encoding='utf8') as f:
  json.dump(result, f, skipkeys=True)

if PLOT_FLAG:
  from pump_opt import plot
  plot.draw.plot_single_diff(output_file, )
