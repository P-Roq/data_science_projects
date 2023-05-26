# Hyperparameters.
class HP:
    initial_vars = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']

    # Data transformation.
    # (....) 

    # Query DataFrame.
    queries = [
        'smoker == 0',
        'bmi > 30 & smoker == 1',
        'bmi > 30 & smoker == 1 & sex == 1',
    ]
    
    # Replace the main dataframe for the rest of the analysis for a queried version.
    queried_df = queries[0]  # is activated when `change_df = True`

    # Visualization hyperparameters.
    hist_cols = initial_vars
    boxplot_cols = ['age', 'bmi', 'children', 'charges']
    heatmap_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
    scatter_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

    # Scatter plot: queried feats vs target.
    queries_scatter = queries
    feat_to_query = 'age'
    target_to_query = 'charges'


    # Container for VIF analysis.
    x_var_container_vif = [
        ['age', 'sex'],
        ['age', 'sex', 'bmi'],
        ['age', 'sex', 'smoker'],
        ['age', 'sex', 'bmi', 'children', 'smoker'],
    ]

    # Regressions preprocessing. 
    split_data_dict = {'rand_state': 5, 'test_size': 0.2}

    # RunRegression hyperparameters.
    x_var_container = [
        ['age', 'sex'],
        ['age', 'sex', 'bmi'],
        ['age', 'sex', 'smoker'],
        ['age', 'sex', 'bmi', 'children', 'smoker'],
    ]

    target_container = ['charges', 'charges', 'charges', 'charges']

    # Section control: choose which sections to run.
    identify_origin_script = True
    data_description_1 = True
    process_data = True
    query_df = True
    change_df = False
    data_description_2 = True
    visualize = True # hist, heat map, and scatter panels
    scatter_query_df = True
    vif_analysis = True
    make_regression = True
    residuals_analysis = False # Run residual analysis.
    residuals_set = 'train' # 'train' or 'test'
    error_comparison = False # Error quality comparison.

    # Choose to store output.
    store_pdf = False
    pdf_version = 1
