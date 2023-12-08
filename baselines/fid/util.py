# Code adapted from https://github.com/facebookresearch/FiD/tree/main.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE-CC-BY-NC-4.0 file in baselines/fid/.

import csv
import json
import logging
import os
import torch

from pathlib import Path

logger = logging.getLogger()


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def write_output(glob_path, output_path):
    files = list(glob_path.glob("*.txt"))
    files.sort()
    with open(output_path, "w") as outfile:
        for path in files:
            with open(path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    outfile.write(line)
            path.unlink()
    glob_path.rmdir()


def save_distributed_dataset(data, opt):
    dir_path = Path(opt.checkpoint_dir) / opt.name
    write_path = dir_path / "tmp_dir"
    write_path.mkdir(exist_ok=True)
    tmp_path = write_path / f"{opt.global_rank}.json"
    with open(tmp_path, "w") as fw:
        json.dump(data, fw)
    if opt.is_distributed:
        torch.distributed.barrier()
    if opt.is_main:
        final_path = dir_path / "dataset_wscores.json"
        logger.info(f"Writing dataset with scores at {final_path}")
        results_path = write_path.glob("*.json")
        alldata = []
        for path in results_path:
            with open(path, "r") as f:
                data = json.load(f)
            alldata.extend(data)
            path.unlink()
        with open(final_path, "w") as fout:
            json.dump(alldata, fout, indent=4)
        write_path.rmdir()


def load_passages(path):
    if not os.path.exists(path):
        logger.info(f"{path} does not exist")
        return
    logger.info(f"Loading passages from: {path}")
    passages = []
    with open(path) as fin:
        reader = csv.reader(fin, delimiter="\t")
        for k, row in enumerate(reader):
            if not row[0] == "id":
                try:
                    passages.append((row[0], row[1], row[2]))
                except IndexError:
                    logger.warning(
                        f"The following input line has not been correctly loaded: {row}"
                    )
    return passages
