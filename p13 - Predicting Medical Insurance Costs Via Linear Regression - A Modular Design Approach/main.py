import subprocess, re, importlib, glob

from src import data_loader
from src import data_transformation as dt
from src import data_visualization as dv
from src import modelling 
from src import statistical_tests as st

# Importing the hyperparameters script by importing the module that starts with 
# the expression 'parameters'.
extract_name = re.findall('./(\w+)', glob.glob('src/para*.py')[0])[0]
p = importlib.import_module(f'src.{extract_name}')

class DataPathClass:
    DATA_PATH = "../p13 - Predicting Medical Insurance Costs Via Linear Regression - A Modular Design Approach/data/insurance.csv"
    PDF_PATH = "../p13 - Predicting Medical Insurance Costs Via Linear Regression - A Modular Design Approach/stored_pdf"
    NOTEBOOK_PATH = "../p13 - Predicting Medical Insurance Costs Via Linear Regression - A Modular Design Approach/output.ipynb"
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

def main() -> None:
    if p.HP.identify_origin_script:
        print(f'Set-up script: {extract_name}.py\n\n')

    ins = data_loader.read_data(DataPathClass.DATA_PATH)

    print('#### Initial Variables:\n') 
    print(ins.info(), '\n')
    print('First rows:\n')     
    print(ins.head())
    print('\n')

    if p.HP.data_description_1:
        print('\nVariable Description Before Data Processing:\n')
        print(ins.describe(), '\n')

    if p.HP.process_data:
        ins = dt.convert_binary(['sex', 'smoker'], ins)
        ins = dt.categorize_var('region', ins)
        ins = dt.log_vars([p.HP.target_container[0]], ins)

        print('\n#### Variables after transformation:\n') 
        print(ins.info(), '\n')
        print('First rows:\n')     
        print(ins.head())

    if p.HP.query_df:
        filter_df = dt.Filtered_DF()
        for query in p.HP.queries:
            filter_df.insert_filtered_df(query, ins)

        if p.HP.change_df == True:
            ins = filter_df.filtered_dict[p.HP.queried_df] 

    if p.HP.data_description_2:
        print('\nVariable Description After Data Processing:\n')
        print(ins.describe(), '\n')

    if p.HP.visualize: 
        dv.histogram_panel(p.HP.hist_cols, ins)
        dv.boxplot_panel(p.HP.boxplot_cols, ins)
        dv.heat_map(p.HP.heatmap_cols, ins)
        dv.scatterplot_panel(
            p.HP.scatter_cols,
            p.HP.target_container[0],
            ins,
            custom_title='Scatter Plots: Features Vs Target'
            )

    if p.HP.scatter_query_df:
        # From the queried DFs available in the dictionary `filtered_dict` pick those to be analyzed
        # and group them in a smaller dictionary - `query_selection`.
        query_selection = {
            key: filter_df.filtered_dict[key] for key in filter_df.filtered_dict if key in p.HP.queries_scatter
            }
        dv.scatter_compare_filtered(
            query_selection,
            p.HP.feat_to_query,
            p.HP.target_to_query,
            )

    if p.HP.vif_analysis:
        print('\n## Analysis of Variance Inflation Factor: \n')
        vif_obj = st.VIF()
        vif_obj.df = ins
        vif_obj.X_vars = p.HP.x_var_container_vif
        vif_obj.store_vif()
        vif_obj.print_vif_container()

    if p.HP.make_regression:
        print('\n#### Regression Results\n\n')
        train, test = dt.split_data(
            df=ins,
            rand_state= p.HP.split_data_dict['rand_state'],
            test_size_= p.HP.split_data_dict['test_size']
            )

        run_reg = modelling.RunRegressions()
        run_reg.train = train
        run_reg.test = test
        
        for x_var_list, target in zip(p.HP.x_var_container, p.HP.target_container):
            run_reg.X_vars.append(x_var_list)
            run_reg.Y.append(target)
        
        run_reg.print_summary()
        run_reg.produce_all_results()
        run_reg.print_all_results(p.HP.residuals_analysis, p.HP.residuals_set)
        # run_reg.flush_results()

        if p.HP.error_comparison:
            print('\n## Error Measurement Comparison\n')
            print(run_reg.compare_error_results())
            print('\n')


    if p.HP.store_pdf:

        # clean_cache_cmd = f'jupyter nbconvert {DataPathClass.NOTEBOOK_PATH} --stdout '
        # clean_cache = subprocess.Popen(clean_cache_cmd, shell=True)

        abs_output_path = f"{DataPathClass.PDF_PATH}/notebook_version_{p.HP.pdf_version}.pdf"
        to_pdf = f"jupyter nbconvert --to pdf --TemplateExporter.exclude_input=True {DataPathClass.NOTEBOOK_PATH} --output-dir {abs_output_path}"
        # to_pdf = f"jupyter nbconvert --to pdf --no-prompt --TemplateExporter.exclude_input=True {DataPathClass.NOTEBOOK_PATH} --output-dir {abs_output_path}"
        save_output = subprocess.Popen(to_pdf, shell=True)


    print('[End Of Report]')

    return 


  