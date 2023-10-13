from sklearn.tree import DecisionTreeRegressor
import argparse
import os
import numpy as np
from sklearn.metrics import root_mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core import Dataset, Workspace


def clean_data(data):
    # all data is numeric so there is no need to one-hot-encode any variable.
    x_df = data.to_pandas_dataframe()
    y_df = x_df.pop("Renta neta media anual de los hogares (Urban Audit)")
    return x_df, y_df


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_depth', type=int, default=20,
                        help="The maximum depth of the tree.")
    parser.add_argument('--min_samples_split', type=float, default=0.1,
                        help="The minimum number of samples required to split an internal node (fraction of total samples)")
    parser.add_argument('--min_samples_leaf', type=float, default=0.1,
                        help="The minimum number of samples required to be at a leaf node (fraction of total samples)")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Maximum Depth:", np.int(args.max_depth))
    run.log("Minimum number of samples per split (% of samples):", np.round(args.min_samples_split,2))
    run.log("Minimum number of samples per leaf node (% of samples):", np.round(args.min_samples_leaf,2))


    subscription_id = '9a8ef160-b36c-4d1c-95d2-b381d53baaa3'
    resource_group = 'rg-bigdatanetworks-uad-pro'
    workspace_name = 'aml-BigDataNetworksuad-pro'

    ws = Workspace(subscription_id, resource_group, workspace_name)
    dataset = Dataset.get_by_name(ws, name='panel_indicadores_distritos_barrios_2022')    
    x, y = clean_data(dataset)

    # TODO: Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=33)

    model = DecisionTreeRegressor(max_depth=args.max_depth, min_samples_split=args.min_samples_split, min_samples_leaf=args.min_samples_leaf).fit(x_train, y_train)

    r2 = model.score(x_test, y_test)
    run.log("r2", np.float(r2))

    joblib.dump(model, 'outputs/model.pkl')

if __name__ == '__main__':
    main()