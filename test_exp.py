# %% [code]
import sys
sys.path.append('../input/iterative-stratification')
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

sys.path.append('../input/moautilsmaster')
import moa_utils

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
warnings.filterwarnings('ignore')

# %% [code]
train_features = pd.read_csv('../input/lish-moa/train_features.csv')
train_targets_scored = pd.read_csv('../input/lish-moa/train_targets_scored.csv')
train_targets_nonscored = pd.read_csv('../input/lish-moa/train_targets_nonscored.csv')

test_features = pd.read_csv('../input/lish-moa/test_features.csv')
sample_submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

GENES = [col for col in train_features.columns if col.startswith('g-')]
CELLS = [col for col in train_features.columns if col.startswith('c-')]

train_features

# %% [code]
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed=42)

# %% [code]
## TODO: apply preprocess before doing the folds (eg: pca)

train = train_features.merge(train_targets_scored, on='sig_id')
test = test_features
target = train[train_targets_scored.columns]

folds = train.copy()
NFOLDS = 5

mskf = MultilabelStratifiedKFold(n_splits=NFOLDS)

for f, (t_idx, v_idx) in enumerate(mskf.split(X=train, y=target)):
    folds.loc[v_idx, 'kfold'] = int(f)

folds['kfold'] = folds['kfold'].astype(int)
folds

# %% [code]
from moa_utils.main import process_data

#target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
#feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
#feature_cols = [c for c in feature_cols if c not in ['kfold','sig_id']]
#len(feature_cols)

# %% [code]
# HyperParameters


from moa_utils.main import run_training

# oof_, predictions_ = run_training({
#     "seed": 42,
#     "fold": 0,
#     "folds": folds,
#     "test": test,
#     "target": target,
#     "feature_cols": feature_cols,
#     "target_cols": target_cols,
#     "batch_size": 128,
#     "num_features": len(feature_cols),
#     "num_targets": len(target_cols),
#     "hidden_size": 512,
#     "device": ('cuda' if torch.cuda.is_available() else 'cpu'),
#     "lr": 1e-3,
#     "weight_decay": 1e-5,
#     "epochs": 25,
#     "early_stopping_steps": 10,
#     "early_stop": False

# })

# %% [code]
# print(oof_.shape)
# print(predictions_.shape)
# %% [code]
import ray
from ray import tune
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import Repeater
from ray.tune.schedulers import AsyncHyperBandScheduler

ray.shutdown()
ray.init(local_mode=True, dashboard_host="0.0.0.0", num_cpus=1) # num_cpus limited the cpu assign to ray, default will use all

from moa_utils.main import set_cols
set_cols(folds, test, target)

# config = {
#              "seed": 42,
#              "batch_size": 128,
# #             "num_features": len(feature_cols),
# #             "num_targets": len(target_cols),
# #             "hidden_size": 512, #tune.grid_search([128, 256, 512, 768]), #tune.grid_search([128, 256, 512, 768]), #128, #tune.uniform(128, 768), #512
#              "device": ('cuda' if torch.cuda.is_available() else 'cpu'),
#              "lr": 1e-3,
#              "weight_decay": 1e-5,
#              "epochs": 25,
#              "early_stopping_steps": 10,
#              "early_stop": False,
#              "is_drop_cp_type": True,
#              #"pca_gens_n_comp": 0, #tune.grid_search([0, 25, 50]),
#              #"pca_cells_n_comp": tune.grid_search([0, 5, 10]),
#              #"pca_cells_n_comp": tune.uniform(0, 10)
#          }

space = {
    "seed": 42,
    "batch_size": 128,
    "device": ('cuda' if torch.cuda.is_available() else 'cpu'),
    "epochs": 40,
    "early_stopping_steps": 10,
    "early_stop": False,
    "is_drop_cp_type": True,
    "pca_cells_n_comp": hp.uniform('pca_cells_n_comp', 0.7, 0.99),
    "pca_gens_n_comp": hp.uniform('pca_gens_n_comp', 0.7, 0.99),
    "network": hp.choice("network", [
        {
            "type": "linear3",
            "hidden_size": hp.quniform("linear3_hidden_size", 128, 768, 1)
        },
        # {
        #     "type": "linear4",
        #     "hidden_size_1": hp.quniform("linear4_hidden_size_1", 128, 768, 1),
        #     "hidden_size_2": hp.quniform("linear4_hidden_size_2", 128, 768, 1)
        # }
    ]),
    "opt": hp.choice("opt", [
        {
            "type": "Adam",
            "lr": hp.choice("Adam_lr", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
            "weight_decay": hp.uniform("Adam_weight_decay", 0, 1e-1),
        },
        {
            "type": "AdamW",
            "lr": hp.choice("AdamW_lr", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
            "weight_decay": hp.uniform("AdamW_weight_decay", 0, 1e-1),
        },
        {
            "type": "SGD",
            "lr": hp.choice("SGD_lr", [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
            "weight_decay": hp.uniform("SGD_weight_decay", 0, 1e-1),
        }
    ]),
}
current_best_params = [
    {
        "seed": 42,
        "batch_size": 128,
        "device": ('cuda' if torch.cuda.is_available() else 'cpu'),
        "epochs": 40,
        "early_stopping_steps": 10,
        "early_stop": False,
        "is_drop_cp_type": True,
        "pca_cells_n_comp": 0.7,
        "pca_gens_n_comp": 0.7,
        "network": {
            "type": "linear3",
            "hidden_size": 512
        },
        "opt": {
            "type": "Adam",
            "lr": 1e-3,
            "weight_decay": 1e-5
        }
    }
]

hyperopt = HyperOptSearch(
    metric="valid_loss", mode="min",
    n_initial_points=5, max_concurrent=1,
    points_to_evaluate=current_best_params,
    space=space)
re_search_alg = Repeater(hyperopt, repeat=NFOLDS)

# from moa_utils.main import set_hyperopt
# set_hyperopt(hyperopt)

ahb = AsyncHyperBandScheduler(
        time_attr="training_iteration",
        metric="valid_loss",
        mode="min",
        grace_period=5,
        max_t=100)

tune.run(run_training,
         # config=config,
         name="hyperopt_run_1",
         local_dir="./ray_results",
         search_alg=re_search_alg,
         scheduler=ahb,
         num_samples=NFOLDS*60,
         #stop={"training_iteration": 5},
         resources_per_trial={"cpu": 1}
        )

hyperopt.save("./hyperopt.cp")
# %% [code]
