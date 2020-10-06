import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from ray import tune

from moa_utils.dataset import MoADataset, TestDataset
from moa_utils.models import linear
from moa_utils import fe

def train_fn(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        #         print(inputs.shape)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss


def valid_fn(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds


def inference_fn(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds


def process_data(data):
    cat_columns = ['cp_time', 'cp_dose', 'cp_type']
    dummy_columns = []
    for c in cat_columns:
        if c in data.columns:
            dummy_columns.append(c)
    data = pd.get_dummies(data, columns=dummy_columns)
    #     data.loc[:, 'cp_time'] = data.loc[:, 'cp_time'].map({24: 0, 48: 1, 72: 2})
    #     data.loc[:, 'cp_dose'] = data.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    # --------------------- Normalize ---------------------
    #     for col in GENES:
    #         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))

    #     for col in CELLS:
    #         data[col] = (data[col]-np.mean(data[col])) / (np.std(data[col]))

    # --------------------- Removing Skewness ---------------------
    #     for col in GENES + CELLS:
    #         if(abs(data[col].skew()) > 0.75):

    #             if(data[col].skew() < 0): # neg-skewness
    #                 data[col] = data[col].max() - data[col] + 1
    #                 data[col] = np.sqrt(data[col])

    #             else:
    #                 data[col] = np.sqrt(data[col])

    return data

folds = None
# feature_cols = None
# target_cols = None
test = None
target = None


def set_cols(_folds, _test, _target):
    global folds
    # global feature_cols
    # global target_cols
    global test
    global target

    folds = _folds
    # feature_cols = fc
    # target_cols = tc
    test =_test
    target = _target

def run_training(c):
    # fold, seed
    seed = c["seed"]
    fold = c["fold"]
    # folds = c["folds"]
    # test = c["test"]
    # target = c["target"]
    global folds
    global test
    global target
    # feature_cols = c["feature_cols"]
    # target_cols = c["target_cols"]
    batch_size = c["batch_size"]
    # num_features = c["num_features"]
    # num_targets = c["num_targets"]
    hidden_size = c["hidden_size"]
    device = c["device"]
    lr = c["lr"]
    weight_decay = c["weight_decay"]
    epochs = c["epochs"]
    early_stopping_steps = c["early_stopping_steps"]
    early_stop = c["early_stop"]

    is_drop_cp_type = False if "is_drop_cp_type" not in c else c["is_drop_cp_type"]
    pca_gens_n_comp = None if "pca_gens_n_comp" not in c else c["pca_gens_n_comp"]
    pca_cells_n_comp = None if "pca_cells_n_comp" not in c else c["pca_cells_n_comp"]

    if is_drop_cp_type:
        folds, test = fe.drop_cp_type(folds, test)

    if pca_gens_n_comp is not None:
        GENES = [col for col in folds.columns if col.startswith('g-')]
        folds, test = fe.fe_pca(folds, test, pca_gens_n_comp, GENES, "g")

    if pca_cells_n_comp is not None:
        CELLS = [col for col in folds.columns if col.startswith('c-')]
        folds, test = fe.fe_pca(folds, test, pca_gens_n_comp, CELLS, "c")

    target_cols = target.drop('sig_id', axis=1).columns.values.tolist()
    feature_cols = [c for c in process_data(folds).columns if c not in target_cols]
    feature_cols = [c for c in feature_cols if c not in ['kfold', 'sig_id']]
    num_features = len(feature_cols)
    num_targets = len(target_cols)

    # seed_everything(seed)

    train = process_data(folds)
    test_ = process_data(test)

    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index

    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)

    x_train, y_train = train_df[feature_cols].values, train_df[target_cols].values
    x_valid, y_valid = valid_df[feature_cols].values, valid_df[target_cols].values

    train_dataset = MoADataset(x_train, y_train)
    valid_dataset = MoADataset(x_valid, y_valid)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = linear.Model1(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=epochs, steps_per_epoch=len(trainloader))

    loss_fn = nn.BCEWithLogitsLoss()

    early_step = 0

    oof = np.zeros((len(train), target.iloc[:, 1:].shape[1]))
    best_loss = np.inf

    for epoch in range(epochs):

        train_loss = train_fn(model, optimizer, scheduler, loss_fn, trainloader, device)
        valid_loss, valid_preds = valid_fn(model, loss_fn, validloader, device)
        print(f"FOLD: {fold}, EPOCH: {epoch}, train_loss: {train_loss}, valid_loss: {valid_loss}")

        tune.report(train_loss=train_loss, valid_loss=valid_loss)

        if valid_loss < best_loss:

            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"FOLD{fold}_.pth")

        elif early_stop:

            early_step += 1
            if (early_step >= early_stopping_steps):
                break

    # --------------------- PREDICTION---------------------
    x_test = test_[feature_cols].values
    testdataset = TestDataset(x_test)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=False)

    model = linear.Model1(
        num_features=num_features,
        num_targets=num_targets,
        hidden_size=hidden_size,
    )

    model.load_state_dict(torch.load(f"FOLD{fold}_.pth"))
    model.to(device)

    predictions = np.zeros((len(test_), target.iloc[:, 1:].shape[1]))
    predictions = inference_fn(model, testloader, device)

    return oof, predictions