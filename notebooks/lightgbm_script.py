# HOME CREDIT DEFAULT RISK COMPETITION
# Most features are created by applying min, max, mean, sum and var functions to grouped tables.
# Little feature selection is done and overfitting might be a problem since many features are related.
# The following key ideas were used:
# - Divide or subtract important features to get rates (like annuity and income)
# - In Bureau Data: create specific features for Active credits and Closed credits
# - In Previous Applications: create specific features for Approved and Refused applications
# - Modularity: one function for each table (except bureau_balance and application_test)
# - One-hot encoding for categorical features
# All tables are joined with the application DF using the SK_ID_CURR key (except bureau_balance).
# You can use LightGBM with KFold or Stratified KFold.

# Update 16/06/2018:
# - Added Payment Rate feature
# - Removed index from features
# - Use standard KFold CV (not stratified)

import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from lightgbm import early_stopping, log_evaluation

warnings.simplefilter(action="ignore", category=FutureWarning)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == "object"]
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv("./src/application_train.csv", nrows=num_rows)
    test_df = pd.read_csv("./src/application_test.csv", nrows=num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = pd.concat([df, test_df], ignore_index=True)
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df["CODE_GENDER"] != "XNA"]

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"]:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df["DAYS_EMPLOYED"].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_CREDIT_PERC"] = df["AMT_INCOME_TOTAL"] / df["AMT_CREDIT"]
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    df["ANNUITY_INCOME_PERC"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_RATE"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    del test_df
    gc.collect()
    return df


"""
# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('./src/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('./src/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    # 1. Extraction des sous-ensembles AVANT le One-Hot Encoding
    active = bureau[bureau['CREDIT_ACTIVE'] == 'Active']
    closed = bureau[bureau['CREDIT_ACTIVE'] == 'Closed']

    # 2. Application du One-Hot Encoding sur le DataFrame principal
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg
"""


def bureau_and_balance(num_rows=None, nan_as_category=True):
    bureau = pd.read_csv("./src/bureau.csv", nrows=num_rows)
    bb = pd.read_csv("./src/bureau_balance.csv", nrows=num_rows)

    # Isolation des sous-ensembles AVANT One-Hot Encoding pour éviter les erreurs de clés
    active = bureau[bureau["CREDIT_ACTIVE"] == "Active"].copy()
    closed = bureau[bureau["CREDIT_ACTIVE"] == "Closed"].copy()

    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    # 1. Bureau balance: Aggregations par SK_ID_BUREAU
    bb_aggregations = {"MONTHS_BALANCE": ["min", "max", "size"]}
    for col in bb_cat:
        bb_aggregations[col] = ["mean"]

    bb_agg = bb.groupby("SK_ID_BUREAU").agg(bb_aggregations)
    bb_agg.columns = pd.Index(
        [e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()]
    )

    # Fusion avec bureau
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
    del bb, bb_agg
    gc.collect()

    # 2. Définition dynamique des agrégations par SK_ID_CURR
    num_aggregations = {
        "DAYS_CREDIT": ["min", "max", "mean", "var"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
    }

    # Inclusions sécurisées des colonnes optionnelles si elles existent réellement dans bureau
    bb_added_cols = [
        "MONTHS_BALANCE_MIN",
        "MONTHS_BALANCE_MAX",
        "MONTHS_BALANCE_SIZE",
    ]
    for col in bb_added_cols:
        if col in bureau.columns:
            num_aggregations[col] = ["mean"]

    cat_aggregations = {}
    for cat in bureau_cat:
        if cat in bureau.columns:
            cat_aggregations[cat] = ["mean"]

    for cat in bb_cat:
        col_name = cat + "_MEAN"
        if col_name in bureau.columns:
            cat_aggregations[col_name] = ["mean"]

    # Aggregation globale bureau par SK_ID_CURR
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        {**num_aggregations, **cat_aggregations}
    )
    bureau_agg.columns = pd.Index(
        ["BURO_" + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()]
    )

    # 3. Aggregations spécifiques pour crédits Actifs
    active_num_aggs = {k: v for k, v in num_aggregations.items() if k in active.columns}
    if not active.empty and active_num_aggs:
        active_agg = active.groupby("SK_ID_CURR").agg(active_num_aggs)
        active_agg.columns = pd.Index(
            ["ACTIVE_" + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()]
        )
        bureau_agg = bureau_agg.join(active_agg, how="left", on="SK_ID_CURR")
        del active_agg
    del active

    # 4. Aggregations spécifiques pour crédits Fermés
    closed_num_aggs = {k: v for k, v in num_aggregations.items() if k in closed.columns}
    if not closed.empty and closed_num_aggs:
        closed_agg = closed.groupby("SK_ID_CURR").agg(closed_num_aggs)
        closed_agg.columns = pd.Index(
            ["CLOSED_" + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()]
        )
        bureau_agg = bureau_agg.join(closed_agg, how="left", on="SK_ID_CURR")
        del closed_agg
    del closed, bureau
    gc.collect()

    return bureau_agg


#############
"""
# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('./src/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']

    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg
"""


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows=None, nan_as_category=True):
    pos = pd.read_csv("./src/POS_CASH_balance.csv", nrows=num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category=True)
    # Features
    aggregations = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
    }
    for cat in cat_cols:
        aggregations[cat] = ["mean"]

    pos_agg = pos.groupby("SK_ID_CURR").agg(aggregations)
    pos_agg.columns = pd.Index(
        ["POS_" + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()]
    )
    # Count pos cash accounts
    pos_agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    del pos
    gc.collect()
    return pos_agg


# Preprocess previous_applications.csv
def previous_applications(num_rows=None, nan_as_category=True):
    prev = pd.read_csv("./src/previous_application.csv", nrows=num_rows)

    # 1. Isolation des sous-ensembles AVANT One-Hot Encoding
    approved = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"].copy()
    refused = prev[prev["NAME_CONTRACT_STATUS"] == "Refused"].copy()

    # 2. Application du One-Hot Encoding sur le DataFrame principal
    prev, cat_cols = one_hot_encoder(prev, nan_as_category=True)

    # Days 365.243 values -> nan (remplacement propre sans inplace=True)
    for col in [
        "DAYS_FIRST_DRAWING",
        "DAYS_FIRST_DUE",
        "DAYS_LAST_DUE_1ST_VERSION",
        "DAYS_LAST_DUE",
        "DAYS_TERMINATION",
    ]:
        prev[col] = prev[col].replace(365243, np.nan)

    # Add feature: value ask / value received percentage
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]

    # Previous applications numeric features
    num_aggregations = {
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean"],
        "APP_CREDIT_PERC": ["min", "max", "mean", "var"],
        "AMT_DOWN_PAYMENT": ["min", "max", "mean"],
        "AMT_GOODS_PRICE": ["min", "max", "mean"],
        "HOUR_APPR_PROCESS_START": ["min", "max", "mean"],
        "RATE_DOWN_PAYMENT": ["min", "max", "mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }

    # Categorical features aggregations
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ["mean"]

    prev_agg = prev.groupby("SK_ID_CURR").agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(
        ["PREV_" + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()]
    )

    # 3. Previous Applications: Approved Applications (sur colonnes numériques existantes)
    approved_num_aggs = {
        k: v for k, v in num_aggregations.items() if k in approved.columns
    }
    if not approved.empty and approved_num_aggs:
        approved_agg = approved.groupby("SK_ID_CURR").agg(approved_num_aggs)
        approved_agg.columns = pd.Index(
            [
                "APPROVED_" + e[0] + "_" + e[1].upper()
                for e in approved_agg.columns.tolist()
            ]
        )
        prev_agg = prev_agg.join(approved_agg, how="left", on="SK_ID_CURR")
        del approved_agg

    # 4. Previous Applications: Refused Applications (sur colonnes numériques existantes)
    refused_num_aggs = {
        k: v for k, v in num_aggregations.items() if k in refused.columns
    }
    if not refused.empty and refused_num_aggs:
        refused_agg = refused.groupby("SK_ID_CURR").agg(refused_num_aggs)
        refused_agg.columns = pd.Index(
            [
                "REFUSED_" + e[0] + "_" + e[1].upper()
                for e in refused_agg.columns.tolist()
            ]
        )
        prev_agg = prev_agg.join(refused_agg, how="left", on="SK_ID_CURR")
        del refused_agg

    del refused, approved, prev
    gc.collect()

    return prev_agg


###########


# Preprocess installments_payments.csv
def installments_payments(num_rows=None, nan_as_category=True):
    ins = pd.read_csv("./src/installments_payments.csv", nrows=num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category=True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins["PAYMENT_PERC"] = ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]
    # Days past due and days before due (no negative values)
    ins["DPD"] = ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]
    ins["DBD"] = ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]
    ins["DPD"] = ins["DPD"].apply(lambda x: x if x > 0 else 0)
    ins["DBD"] = ins["DBD"].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        "NUM_INSTALMENT_VERSION": ["nunique"],
        "DPD": ["max", "mean", "sum"],
        "DBD": ["max", "mean", "sum"],
        "PAYMENT_PERC": ["max", "mean", "sum", "var"],
        "PAYMENT_DIFF": ["max", "mean", "sum", "var"],
        "AMT_INSTALMENT": ["max", "mean", "sum"],
        "AMT_PAYMENT": ["min", "max", "mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean", "sum"],
    }
    for cat in cat_cols:
        aggregations[cat] = ["mean"]
    ins_agg = ins.groupby("SK_ID_CURR").agg(aggregations)
    ins_agg.columns = pd.Index(
        ["INSTAL_" + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()]
    )
    # Count installments accounts
    ins_agg["INSTAL_COUNT"] = ins.groupby("SK_ID_CURR").size()
    del ins
    gc.collect()
    return ins_agg


"""
# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('./src/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg
"""


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows=None, nan_as_category=True):
    cc = pd.read_csv("./src/credit_card_balance.csv", nrows=num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category=True)

    # 1. Isolation de la clé de regroupement
    # sk_id_curr = cc["SK_ID_CURR"]

    # 2. Suppression de SK_ID_PREV car inutile pour les agrégations finales
    if "SK_ID_PREV" in cc.columns:
        cc.drop(["SK_ID_PREV"], axis=1, inplace=True)

    # 3. Filtrage stricte sur les colonnes numériques uniquement
    numeric_cols = cc.select_dtypes(include=[np.number]).columns.tolist()

    # 4. Aggregations uniquement sur le sous-ensemble numérique
    cc_agg = (
        cc[numeric_cols]
        .groupby("SK_ID_CURR")
        .agg(
            [
                "min",
                "max",
                "mean",
                "sum",
                "var",
            ]
        )
    )

    # Renommage à plat des colonnes MultiIndex
    cc_agg.columns = pd.Index(
        ["CC_" + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()]
    )

    # Nombre d'enregistrements par client
    cc_agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()

    del cc
    gc.collect()

    return cc_agg


######
# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
"""
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df
"""


def kfold_lightgbm(df, num_folds, stratified=False, debug=False):
    # 1. Copie explicite du DataFrame pour éviter les conflits de portée (UnboundLocalError)
    data = df.copy()

    # 2. Encodage des colonnes catégorielles sur la copie 'data'
    cat_cols = [
        col
        for col in data.columns
        if data[col].dtype == "object" or data[col].dtype == "string"
    ]
    for col in cat_cols:
        data[col] = data[col].astype("category").cat.codes

    # 3. Séparation train / test
    train_df = data[data["TARGET"].notnull()].copy()
    test_df = data[data["TARGET"].isnull()].copy()

    print(
        "Starting LightGBM. Train shape: {}, test shape: {}".format(
            train_df.shape, test_df.shape
        )
    )

    del data, df
    gc.collect()

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits=num_folds, shuffle=True, random_state=1001)

    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [
        f
        for f in train_df.columns
        if f not in ["TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"]
    ]

    for n_fold, (train_idx, valid_idx) in enumerate(
        folds.split(train_df[feats], train_df["TARGET"])
    ):
        train_x, train_y = (
            train_df[feats].iloc[train_idx],
            train_df["TARGET"].iloc[train_idx],
        )
        valid_x, valid_y = (
            train_df[feats].iloc[valid_idx],
            train_df["TARGET"].iloc[valid_idx],
        )

        clf = LGBMClassifier(
            nthread=-1,  # Utilise tous les cœurs CPU disponibles (-1 au lieu de 4)
            n_estimators=500,  # Au lieu de 10000
            learning_rate=0.1,  # Au lieu de 0.02
            num_leaves=31,
            max_depth=6,
            verbose=-1,
        )

        clf.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_metric="auc",
            callbacks=[
                early_stopping(stopping_rounds=200),
                log_evaluation(period=200),
            ],
        )

        # ...existing code...

        feats = [
            f
            for f in train_df.columns
            if f
            not in [
                "TARGET",
                "SK_ID_CURR",
                "SK_ID_BUREAU",
                "SK_ID_PREV",
                "index",
            ]
        ]

        # Sauvegarde du schéma réellement utilisé par LightGBM
        import joblib
        from pathlib import Path

        model_dir = Path(__file__).resolve().parent.parent / "models"
        model_dir.mkdir(exist_ok=True)

        joblib.dump(feats, model_dir / "feature_names.joblib")

        # ...existing code...

        oof_preds[valid_idx] = clf.predict_proba(
            valid_x, num_iteration=clf.best_iteration_
        )[:, 1]
        sub_preds += (
            clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1]
            / folds.n_splits
        )

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat(
            [feature_importance_df, fold_importance_df], axis=0
        )

        print(
            "Fold %2d AUC : %.6f"
            % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx]))
        )
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print("Full AUC score %.6f" % roc_auc_score(train_df["TARGET"], oof_preds))

    if not debug:
        test_df["TARGET"] = sub_preds
        test_df[["SK_ID_CURR", "TARGET"]].to_csv(submission_file_name, index=False)

    # Dans kfold_lightgbm():
    if not debug:
        test_df["TARGET"] = sub_preds
        test_df[["SK_ID_CURR", "TARGET"]].to_csv(submission_file_name, index=False)

        # Exporter également la feature importance en CSV pour ingestion SQLite
        mean_fi = (
            feature_importance_df.groupby("feature")["importance"].mean().reset_index()
        )
        mean_fi.to_csv("feature_importance.csv", index=False)

    display_importances(feature_importance_df)
    return feature_importance_df


###
# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = (
        feature_importance_df_[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:40]
        .index
    )
    best_features = feature_importance_df_.loc[
        feature_importance_df_.feature.isin(cols)
    ]
    plt.figure(figsize=(8, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=best_features.sort_values(by="importance", ascending=False),
    )
    plt.title("LightGBM Features (avg over folds)")
    plt.tight_layout()
    plt.savefig("lgbm_importances01.png")


def main(debug=False):
    num_rows = 30000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how="left", on="SK_ID_CURR")
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how="left", on="SK_ID_CURR")
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how="left", on="SK_ID_CURR")
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how="left", on="SK_ID_CURR")
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how="left", on="SK_ID_CURR")
        del cc
        gc.collect()
    # with timer("Run LightGBM with kfold"):
    # feat_importance = kfold_lightgbm(df, num_folds=5, stratified=False, debug=debug)


if __name__ == "__main__":
    submission_file_name = "submission_kernel02.csv"
    with timer("Full model run"):
        main()


"""
. Sous-échantillonner le jeu de données (subsample)Au lieu de charger l'intégralité des lignes (plus de 300 000 entrées pour application_train), réduisez la quantité de données chargées avec le paramètre nrows ou en effectuant un subsampling aléatoire. L'importance des variables reste très stable même sur un échantillon réduit à 10 % ou 20 %.Dans la fonction main(), passez debug = True ou définissez un paramètre num_rows = 30000 (au lieu de None) lors de l'appel aux fonctions de prétraitement (application_train_test, bureau_and_balance, etc.).  Python# Dans main()
num_rows = 30000  # Réduit considérablement le temps de lecture et d'agrégation
2. Réduire le nombre de foldsActuellement, votre script exécute une boucle KFold avec num_folds = 10. Cela signifie que LightGBM entraîne 10 modèles complets séquentiellement.  Passez num_folds = 2 ou 3 dans l'appel de kfold_lightgbm pour votre phase d'analyse/exploratoire. Vous diviserez le temps de calcul du modèle par 3 à 5.  Python# Dans main()
feat_importance = kfold_lightgbm(
    df, num_folds=2, stratified=False, debug=True
)
3. Ajuster les hyperparamètres de LightGBM pour la vitesseDans la fonction kfold_lightgbm, modifiez les paramètres de LGBMClassifier pour privilégier la vitesse d'exécution :n_estimators : Baissez-le de 10000 à 300 ou 500.  learning_rate : Augmentez-le de 0.02 à 0.1. Un learning rate plus élevé permet au modèle de converger en beaucoup moins d'arbres.  max_depth / num_leaves : Réduisez légèrement la profondeur pour accélérer la construction des arbres.Pythonclf = LGBMClassifier(
    nthread=-1,  # Utilise tous les cœurs CPU disponibles (-1 au lieu de 4)
    n_estimators=500,  # Au lieu de 10000
    learning_rate=0.1,  # Au lieu de 0.02
    num_leaves=31,
    max_depth=6,
    verbose=-1,
)

"""
