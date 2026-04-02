#!/usr/bin/env python3
"""Stage2 runner for Alpha158 practice with explicit time-decay reweighting.

This script mirrors the Qlib qrun workflow but instantiates a custom
TimeDecayReweighter so we can tune the half-life without modifying the
installed qlib package.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from qlib.config import C
from qlib.constant import REG_CN
from qlib.data.dataset import Dataset
from qlib.log import get_module_logger
from qlib.model.base import Model
from qlib.utils import auto_filter_kwargs, fill_placeholder, flatten_dict, init_instance_by_config
from qlib.workflow import R

from time_decay_reweighter import TimeDecayReweighter

logger = get_module_logger("stage2_practice")


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _init_qlib(qlib_init: dict[str, Any], uri_folder: str) -> None:
    import qlib

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
    qlib.init(**qlib_init, exp_manager=exp_manager)


def _log_task_info(task_config: dict[str, Any]) -> None:
    R.log_params(**flatten_dict(task_config))
    R.save_objects(**{"task": task_config})
    R.set_tags(**{"hostname": os.uname().nodename})


def _exe_task(task_config: dict[str, Any], reweighter: TimeDecayReweighter) -> None:
    rec = R.get_recorder()
    model: Model = init_instance_by_config(task_config["model"], accept_types=Model)
    dataset: Dataset = init_instance_by_config(task_config["dataset"], accept_types=Dataset)

    auto_filter_kwargs(model.fit)(dataset, reweighter=reweighter)
    R.save_objects(**{"params.pkl": model})
    dataset.config(dump_all=False, recursive=True)
    R.save_objects(**{"dataset": dataset})

    placehorder_value = {"<MODEL>": model, "<DATASET>": dataset}
    task_config = fill_placeholder(task_config, placehorder_value)
    records = task_config.get("record", [])
    if isinstance(records, dict):
        records = [records]
    for record in records:
        r = init_instance_by_config(
            record,
            recorder=rec,
            default_module="qlib.workflow.record_temp",
            try_kwargs={"model": model, "dataset": dataset},
        )
        r.generate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Generated practice YAML")
    ap.add_argument("--experiment-name", required=True)
    ap.add_argument("--uri-folder", default="mlruns")
    ap.add_argument("--half-life", type=int, default=252)
    ap.add_argument("--floor", type=float, default=0.2)
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    qlib_init = cfg.get("qlib_init", {})
    task = cfg.get("task", {})

    if args.half_life <= 0:
        raise ValueError("half-life must be positive")
    if not (0 < args.floor <= 1):
        raise ValueError("floor must be in (0, 1]")

    _init_qlib(qlib_init, args.uri_folder)

    reweighter = TimeDecayReweighter(half_life=args.half_life, floor=args.floor)

    with R.start(experiment_name=args.experiment_name):
        _log_task_info(task)
        _exe_task(task, reweighter=reweighter)


if __name__ == "__main__":
    main()
