from scipy.stats import shapiro as shapiro_test
from scipy.stats import normaltest as dagostino_test 
from scipy.stats import kstest as kol_smir_test 
from scipy.stats import jarque_bera as jarque_bera_test

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

import pandas as pd
import numpy as np

def normality_tests(residuals):

    normality_tests = {
        'Shapiro-Wilk': shapiro_test(residuals),
        "D'Agostino's": dagostino_test(residuals),
        'Kolmogorov-Smirnov': kol_smir_test(residuals, 'norm'),
        "Jarque-Bera": jarque_bera_test(residuals),
    }

    for key in normality_tests:
        stat = round(normality_tests[key][0], 4)
        p_value = round(normality_tests[key][1], 4)

        print(f"Test: {key}")
        print(f'    - Statistic: {stat}, p-value: {p_value}')

    return

class VIF:
    def __init__(self):
        self.df: pd.core.frame.DataFrame = None
        self.X_vars: list = []
        self.vif_container: list = []

    def get_vif(self, df, feats):

        df = df[feats]

        numerical = df[[col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]]

        X = add_constant(numerical)

        vif_series = (
            pd.Series(
                [variance_inflation_factor(X.values, i) for i in range(X.shape[1])], 
                index=X.columns,
                name='VIF',
                )
                .drop('const')
                .sort_values()
            )
        
        return vif_series

    def store_vif(self):
        for feats in self.X_vars:
            vif_series = self.get_vif(self.df, feats)
            self.vif_container.append(vif_series)

        return

    def print_vif_container(self):
        for i, vif_series in enumerate(self.vif_container):
            print(f'Features group: {i+1}\n')
            print(vif_series)
            print('\n------------------------------------\n')

        return