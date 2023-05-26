import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def find_number_of_rows(nr_of_axes):
    """Support function that takes the number of axis/graphs to include in the panel and 
    calculates the number of necessary rows to include in the panel with 3 columns. 
    Example: if we panel to include 8 graphs in the panel the number of rows to be 
    returned are 3 so that the panel is a 3x3 (9 total: 8 + 1 omitted).
    """
    ranges = pd.interval_range(start=0, end=60, freq=3, closed='right')

    for i, range_ in enumerate(ranges):
        if nr_of_axes in range_:
            number_of_rows = i+1

    return number_of_rows  


# Histograms panel.
def histogram_panel(var_list: list, df: pd.core.frame.DataFrame):
    
    df = df[var_list]
    number_of_graphs = len(var_list)
    rows = find_number_of_rows(number_of_graphs)
    columns = 3
    max_plots  = rows*columns
    
    if max_plots < number_of_graphs:
        return 'Error: The number of features must fit the the number of plots in the matrix: `rows*columns >= var_list`'

    fig, ax = plt.subplots(rows, columns, figsize=(columns*3.5, rows*3.5))

    plt.suptitle(
        'Histograms',
        size=20,
        y=1
        )

    ax = ax.reshape(-1, 1)

    for i in range(0, max_plots):
        if i < number_of_graphs:
            ax_ = ax[i, 0]
            col = df.columns[i]
            ax_.set_title(col)
            ax_.hist(df[col], density=True, edgecolor = "black")
            ax_.grid(axis='y')
            ax_.set_axisbelow(True)

        else:
            ax_ = ax[i, 0]
            ax_.set_axis_off()

    plt.tight_layout()
    plt.show()

# Box plot panel.
def boxplot_panel(var_list: list, df: pd.core.frame.DataFrame):

    df = df[var_list]
    # Leaving only numerical columns.
    df = df[[col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]]

    number_of_graphs = len(df.columns)
    rows = find_number_of_rows(number_of_graphs)
    columns = 3
    max_plots  = rows*columns
    
    if max_plots < number_of_graphs:
        return 'Error: The number of features must fit the the number of plots in the matrix: `rows*columns >= total numerical columns`'

    fig, ax = plt.subplots(rows, columns, figsize=(columns*3.0, rows*4.5))

    plt.suptitle(
        'Box plots',
        size=20,
        y=1
        )

    ax = ax.reshape(-1, 1)

    for i in range(0, max_plots):
        if i < number_of_graphs:
            ax_ = ax[i, 0]
            col = df.columns[i]
            ax_.set_title(col)
            ax_.boxplot(df[col])
            ax_.grid(axis='y')
            ax_.set_axisbelow(True)

        else:
            ax_ = ax[i, 0]
            ax_.set_axis_off()

    plt.tight_layout()
    plt.show()



# Correlation heatmap.
def heat_map(feats: list, df: pd.core.frame.DataFrame):
    
    corr = df[feats].corr(method='pearson', numeric_only=True)

    # Draw a heatmap with the numeric values in each cell.
    fig, ax = plt.subplots(figsize=(6, 6))

    plt.title('Pearson Correlation Heatmap', size=15)

    # Getting the Upper Triangle of the co-relation matrix.
    matrix = np.triu(corr)

    sns.heatmap(corr,
                annot=True,
                linewidths=.5,
                mask=matrix,
                ax=ax)

    plt.show()


# Scatter plots panel.
def scatterplot_panel(
    var_list: list,
    target_var: str,
    df: pd.core.frame.DataFrame,
    custom_title: str,
    ):
    """Choose a DataFrame, a list of independent variables, a target variable, and the configuration 
    of the panel: rows x columns; to produce a pre-configured panel of scatterplots.
    """
    
    target = df[target_var]
    df = df[var_list]
    number_of_graphs = len(var_list)
    rows = find_number_of_rows(number_of_graphs)
    columns = 3
    max_plots  = rows*columns
    
    if max_plots < number_of_graphs:
        return 'Error: The number of features must fit the the number of plots in the matrix: `rows*columns >= var_list`'

    fig, ax = plt.subplots(rows, columns, figsize=(columns*3.5, rows*3.5))

    plt.suptitle(custom_title, size=20, y=1)

    ax = ax.reshape(-1, 1)

    for i in range(0, max_plots):
        if i < number_of_graphs:
            ax_ = ax[i, 0]
            col = df.columns[i]
            ax_.set_title(col)
            ax_.scatter(df[col], target)
            ax_.grid()
            ax_.set_axisbelow(True)

        else:
            ax_ = ax[i, 0]
            ax_.set_axis_off()

    plt.tight_layout()
    plt.show()


# Scatter plot - queried/filtered dataframes.
def scatter_compare_filtered(
    dic_filtered_dfs: dict,
    feature: str,
    target: str
    ):

    # plt.style.use('classic')

    colors = ['purple', 'blue', 'green', 'orange', 'black', 'yellow',]

    fig, ax = plt.subplots(figsize=(4, 4))

    plt.suptitle(f'Queried Feature: {feature} Vs Target: {target}')

    for i, key in enumerate(dic_filtered_dfs):
        df = dic_filtered_dfs[key]
        ax.scatter(df[feature], df[target], color=colors[i], label=key)
        ax.grid()
        ax.set_axisbelow(True)

    plt.legend(bbox_to_anchor=[1.0, 1.0])
    plt.xlabel(feature)
    plt.ylabel(target)
    
    # plt.tight_layout()
    plt.show()
    
    return  


# Scatter plots panel for residuals (train and test) vs target.
def resid_visual_analysis_1(
    residuals: pd.core.series.Series,
    prediction: pd.core.series.Series,
    target: str,
    df,
    residuals_set: str,
):

    fig, ax = plt.subplots(1, 2, figsize=(2*3.5, 1*3.5))

    plt.suptitle(
        f'Residuals: {residuals_set}',
        size=15,
        y=1
        )

    # Residuals vs target.
    ax[0].scatter(residuals, df[target])
    ax[0].set_title('Residuals Vs Target Variable')
    ax[0].set_ylabel('residuals')
    ax[0].set_xlabel(target)
    ax[0].grid()
    ax[0].set_axisbelow(True)

    # Predicted Vs Observed. 
    ax[1].scatter(prediction, df[target])
    ax[1].set_title('Predicted Vs Observed')
    ax[1].set_xlabel('Predicted')
    ax[1].set_ylabel('Observed')
    ax[1].grid()
    ax[1].set_axisbelow(True)

    plt.tight_layout()
    plt.show()


def resid_visual_analysis_2(residuals: pd.core.series.Series, target_series: pd.core.series.Series):

    # Plots for normal distribution.

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # ax1
    ax1.set_title('Residual Vs Predicted Values', size=15)
    ax1.scatter(target_series, residuals)
    ax1.hlines(0, xmin=target_series.min(), xmax=target_series.max(), colors='red')
    ax1.set_ylabel('Residuals')
    ax1.set_xlabel('Predicted Values')
    ax1.grid()
    ax1.set_axisbelow(True)

    # ax2
    ax2.set_title('Histogram: Distribution Of Residuals', size=15)
    sns.histplot(residuals, kde=True, color='blue', ax=ax2)
    ax2.grid()
    ax2.set_axisbelow(True)


    # ax3
    ax3.set_title('Disparity Between Residuals Quantiles And Normally Distributed Quantiles', size=15)
    sm.ProbPlot(residuals).qqplot(line='s', ax=ax3)
    ax3.grid()
    ax3.set_axisbelow(True)

    
    plt.tight_layout()
    plt.show()