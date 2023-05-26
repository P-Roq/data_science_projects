import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys

from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2

sys.path.append('../src')
from src import data_visualization as dv
from src.statistical_tests import *

# Support function that print regression results contained in a dictionary (the output of `reg_results`).
def result_printer(results: dict) -> None:
    for key in results:
        if key != 'ols':
            print(f'{key} = {results[key]}')
    return

class RunRegressions:
    def __init__(self):
        self.train = None # dataframe
        self.test = None # dataframe
        self.X_vars = []
        self.Y = []
        self.results_store = []


    # Produce and store regression results for one experiment.
    def reg_results(self, feats: list, target: str, train: pd.core.frame.DataFrame, test: pd.core.frame.DataFrame) -> dict:
        
        # Fit.
        X_train = train[feats]
        X_train_w_const = sm.add_constant(X_train, prepend=True)
        y_train = train[target].values.reshape(-1, 1)

        est = sm.OLS(y_train, X_train_w_const.values)
        ols = est.fit()
        
        # Predict.
        X_test = test[feats]
        X_test_w_const = sm.add_constant(X_test, prepend=True)
        y_test = test[target].values.reshape(-1, 1)

        prediction_train = ols.predict(X_train_w_const.values)
        prediction_test = ols.predict(X_test_w_const.values)
        
        # Error.
        mse_ = round(mse(y_test, prediction_test), 2)
        rmse_ = round(np.sqrt(mse_), 2)

        # Residuals
        resids_train = train[target] - prediction_train
        resids_test = test[target] - prediction_test

        resids_train.name = 'resids_train'
        resids_test.name = 'resids_test'

        return {
            'feats': feats,
            'target': target,
            'ols': est.fit(),
            'summary': ols.summary2(),
            'mse': mse_,
            'rmse': rmse_,
            'residuals_train': resids_train, 
            'residuals_test': resids_test,
            'prediction_train': prediction_train,
            'prediction_test': prediction_test,
            }

    def produce_specific_result(store_index: int):
        result = self.reg_results(x_vars[store_index], target[store_index], self.train, self.test)
        self.results_store.append(result)

        return


    def produce_all_results(self):
        for x_vars, target in zip(self.X_vars, self.Y):
            result = self.reg_results(x_vars, target, self.train, self.test)
            self.results_store.append(result)

        return

    def print_summary(self):
        print('## Summary:\n')
        for i, tuple_ in enumerate(zip(self.Y, self.X_vars)):
            print(f'Regression Nr: {i+1}:')
            print(f'    - Target: {tuple_[0]}')
            print(f'    - Explanatory Variables: {tuple_[1]}\n')
        
        return

    def print_specific_result(store_index: int):
        return result_printer(self.results_store[store_index: int])      


    def print_all_results(self, residuals_analysis: bool, residuals_set: str):

        for i, result_dict in enumerate(self.results_store):

            #### Basic information.
            print(f'### Regression number: {i+1}\n')
            print(f'Target variable (Y): {self.Y[i]}\n')
            print('Explanatory Variables:\n')
            for idx, feat in enumerate(self.X_vars[i]):
                print(f"    - x{idx+1}: {feat}")

            #### Regression summary.
            print('\n', result_dict['summary'], '\n')

            #### Error / predictor quality.
            print('## Error measurement:\n')
            print(f"MSE: {result_dict['mse']}")
            print(f"RMSE: {result_dict['rmse']}")

            #### Residuals analysis.
            if residuals_analysis:

                if residuals_set == 'train':
                    df_resids = self.train.copy()
                if residuals_set == 'test':
                    df_resids = self.test.copy()

                print(f'\n## Residuals Analysis for the {residuals_set} set.\n')

                normality_tests(result_dict[f'residuals_{residuals_set}'])

                # Residuals vs target.            
                dv.resid_visual_analysis_1(
                    result_dict[f'residuals_{residuals_set}'],
                    result_dict[f'prediction_{residuals_set}'],
                    result_dict['target'],
                    df_resids,
                    residuals_set,
                    )

                dv.resid_visual_analysis_2(
                    result_dict[f'residuals_{residuals_set}'],
                    self.train[result_dict['target']],
                    )


            print('\n')
            if i == len(self.results_store) - 1:
                print('** [No more experiments] **\n')
            

        return

    def compare_error_results(self):

        compare = pd.DataFrame(
            columns=['mse', 'rmse'],
            index=[
                ', '.join(self.results_store[i]['feats']) for i, val in enumerate(self.results_store)
                ]
        )

        for i, result_dict in enumerate(self.results_store):
            compare.iloc[i, :] = [result_dict['mse'], result_dict['rmse']]

        compare = compare.sort_values(by='mse', ascending=True)

        return compare

    # def flush_results():
    #     self.X_vars = []
    #     self.Y = []
    #     self.results_store = []