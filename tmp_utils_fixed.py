import random
import re
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.utils.env import QTDockerEnv


def generate_data_folder_from_qlib():
    template_path = Path(__file__).parent / factor_data_template
    qtde = QTDockerEnv()
    qtde.prepare()

    execute_log = qtde.check_output(
        local_path=str(template_path),
        entry=python
