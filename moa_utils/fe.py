import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

"""
    output features name will be changed to 0, 1, 2, ...
"""
def variance_thredshold(train_features: pd.DataFrame, test_features: pd.DataFrame, threshold=0.5):

    var_thresh = VarianceThreshold(threshold=threshold)
    data = train_features.append(test_features)
    data_transformed = var_thresh.fit_transform(data.iloc[:, 4:])

    train_features_transformed = data_transformed[ : train_features.shape[0]]
    test_features_transformed = data_transformed[-test_features.shape[0] : ]\

    train_features = pd.DataFrame(train_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                  columns=['sig_id','cp_type','cp_time','cp_dose'])

    train_features = pd.concat([train_features, pd.DataFrame(train_features_transformed)], axis=1)

    test_features = pd.DataFrame(test_features[['sig_id','cp_type','cp_time','cp_dose']].values.reshape(-1, 4),\
                                 columns=['sig_id','cp_type','cp_time','cp_dose'])

    test_features = pd.concat([test_features, pd.DataFrame(test_features_transformed)], axis=1)

    return train_features, test_features


def drop_cp_type(train: pd.DataFrame, test_features: pd.DataFrame):
    train = train[train['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)

    train = train.drop('cp_type', axis=1)
    test = test.drop('cp_type', axis=1)

    return train, test


# required run before VarianceThreshold or changed column name
def binning(train, test, cols, bins=3):
    for col in cols:
        train.loc[:, f'{col}_bin'] = pd.cut(train[col], bins=3, labels=False)
        test.loc[:, f'{col}_bin'] = pd.cut(test[col], bins=3, labels=False)

    return train, test


# need to run before VarianceThreshold
def fe_pca(train, test, cols, col_label, n_comp=50, random_state=42):
    data = pd.concat([pd.DataFrame(train[cols]), pd.DataFrame(test[cols])])
    data2 = (PCA(n_components=n_comp, random_state=random_state).fit_transform(data[cols]))
    train2 = data2[:train.shape[0]];
    test2 = data2[train.shape[0]:]

    train2 = pd.DataFrame(train2, columns=[f'pca_{col_label}-{i}' for i in range(n_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_{col_label}-{i}' for i in range(n_comp)])

    # drop_cols = [f'c-{i}' for i in range(n_comp, len(cols))]
    train = train.drop(cols, axis=1)
    test = test.drop(cols, axis=1)
    train = pd.concat((train, train2), axis=1)
    test = pd.concat((test, test2), axis=1)

    return train, test
