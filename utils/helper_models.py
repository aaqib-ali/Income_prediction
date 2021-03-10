import pandas as pd
import dill as pickle

# sklearn
from sklearn.model_selection import train_test_split

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter

# sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
import scikitplot.metrics as skplt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
# from this project
import utils.common as common


# Function to calculate missing values by column
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Missing Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Missing Values', ascending=False).round(3)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1])
          + " columns.\n""There are " + str(mis_val_table_ren_columns.shape[0])
          + " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def split_dataset(df):

    print('Random Spliting....')

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['income'], shuffle=True)

    print("Split train set", df_train.shape[0], "test set", df_test.shape[0])

    return df_train, df_test

def read_features_name():

    # model settings
    target_name = 'income'

    print('Read Features Name')

    # List of selected/best features

    categorical_features = ['occupation', 'relationship', 'race']
    numeric_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

    return categorical_features, numeric_features, target_name

def read_parameters(name):
    # name is the json file name without .json
    # the file is expected to be found in the folder model_parameters
    print(os.path.join("model_parameters", name + "params.json"))

    with open(os.path.join("model_parameters", name + "params.json")) as fjson:
        json_content = json.load(fjson)
    try:
        return json_content["params"]
    except KeyError:
        print("Key params not found", json_content)

def plot_roc_curve_and_save(df_test_Y, probability, f_name='Roc Curve', out_dir="."):
    # plot roc
    plot_micro = False
    plot_macro = False

    print('Test AUC: %1.4f' % roc_auc_score(df_test_Y, probability[:, 1]))
    plt.figure(f_name)
    ax = plt.gca()

    skplt.plot_roc(df_test_Y, probability, title=f_name, plot_micro=plot_micro, plot_macro=plot_macro, ax=ax)

    common.save_fig(f_name, os.path.join(out_dir))


def fill_confusion_matrix_and_save(df_test_Y, prediction, f_name="Confusion matrix", out_dir=".", normalize=True):

    labels = unique_labels(df_test_Y, prediction)

    cm = confusion_matrix(df_test_Y, prediction)

    print('Confusion Matrix', cm)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    y_label = 'True label'
    x_label = 'Predicted label'

    plt.figure(f_name)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=f_name,
           ylabel=y_label,
           xlabel=x_label)
    common.save_fig(f_name, out_dir)


def donwsample_trainset(train_data_df: pd.DataFrame, target_name, RSEED):
    # split by class
    class_0: pd.DataFrame = train_data_df[train_data_df[target_name] == 0]
    class_1: pd.DataFrame = train_data_df[train_data_df[target_name] == 1]

    c0_length = class_0.shape[0]
    c1_length = class_1.shape[0]
    print('Train Targets shape %s' % Counter(train_data_df[target_name]))

    # downsample and make the train set with equal ratio of No_VT and VT class
    # class 1 is scarce
    frac = (0.50 * c1_length) / (c0_length - c0_length * 0.50)
    print("Downsampling the classes to {} of class 0 and {} of class 1".format(frac * c0_length, c1_length))
    # sample according to fraction, concat with other class and reshuffle
    train_data_df = pd.concat([class_0.sample(frac=frac, replace=False, random_state=RSEED), class_1]).sample(frac=1)

    print('Train Targets shape %s' % Counter(train_data_df[target_name]))

    #print(train_data_df.head())

    return train_data_df

def plot_feature_importance_and_save(pipeline, categorical_features, numeric_features, top_num=5, f_name="Feature Importance", out_dir="."):
    ohe = (pipeline.named_steps['preprocess']
        .named_transformers_['cat']
        .named_steps['onehot'])

    feature_names = ohe.get_feature_names(input_features=categorical_features)
    feature_names = np.r_[feature_names, numeric_features]

    tree_feature_importances = (
        pipeline.named_steps['classifier'].feature_importances_)

    feat_imp = pd.DataFrame({'importance': tree_feature_importances})
    feat_imp['feature'] = feature_names
    feat_imp.sort_values(by='importance', ascending=False, inplace=True)

    # Select top 5 features
    feat_imp = feat_imp.iloc[:top_num]

    feat_imp.sort_values(by='importance', inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)
    plt.figure(f_name)
    ax = plt.gca()
    ax.autoscale(enable=True)
    feat_imp.plot.barh(title="Feature Importances", figsize=(20,10), ax=ax)

    plt.xlabel('Feature Importance Score')
    common.save_fig(f_name, out_dir)

    print('Feature importance plot is saved to the Directory...')

def save_pipeline(pipeline):

    filename = 'model_final.pk'

    with open(os.path.join("models", filename), 'wb') as file:
        pickle.dump(pipeline, file)

    print('Pipeline Saved Successfully to the root directory')

def load_pipeline():

    filename = 'model_final.pk'

    # Open saved model, and directly make the prediction with new data
    with open(os.path.join("models", filename), 'rb') as f:
        loaded_pipeline = pickle.load(f)

    return loaded_pipeline






