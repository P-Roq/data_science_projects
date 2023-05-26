# Project Name: Predicting Insurance Costs Via Linear Regression - A Modular Design Approach
  

## Details:


1. The overall purpose of the application:

    - This application pre-configures a linear regression workflow which aims at predicting medical insurance costs of patients, by allowing the use of quick set-up hyperparameter scripts that instruct the app to produce and customize a comprehensive statistical analysis report, that can be exported into a PDF document. This is possible by designing the framework of the analysis using a modular approach instead of a vertical functional analysis. The major purpose of this app is to create a version that can eventually be modified to a level of abstraction that allows to use apply it to any data set or theme instead of having it tied to the medical insurance costs data set.  


2. How the application can be used to address specific needs or objectives:

    - A problem arises when a researcher wants to try several approaches to the various stages of the analysis before it settles into a final configuration that wishes to interpret and explain - e.g. changing the variables to be analyzed; determining how many experimental regressions to perform; omit or ignore one or more sections of the analysis, but has no way simple way to quickly produce or reproduce a modified version of the current analysis. This app aims at allowing the researcher to perform several exploratory and statistical tasks, by changing only argument values in a dedicated control script without having to constantly change the core structure of the framework.
      

3. The main features and functions of the application:

    - Produce a full or partial linear regression analysis workflow in a Jupyter Notebook.
        - Chose what sections (stages or sub-sections of the analysis) to be performed.
        - Store multiple reports by either storing the set-up script or by exporting the report into a PDF file.

    - It allows to change a number of arguments at several stages of the data pipeline that modify aspects of:
        - Exploratory data analysis / data visualization.
        - Data processing and splitting.
        - Regression analysis and sub-component analysis.

    - The quick set-up script is a module identified by the initial expression 'parameters', e.g. 'parameters_1.py'. The app will try to identify a script with that expression and load the module (if more than one script, the one loaded is the first to be extracted using the `glob` function). The ideal workflow is to copy-paste one script from the folder 'quick_setups' into the 'src' file; as mentioned, the script should start with the expression 'parameter' and preferably, there should be only one script of that type located in 'src'.  


4. The target audience the application is designed for:

    - The app is targeted to students/researchers/practitioners within the econometrics, data analysis and  data science fields. 


5. The value the application provides to users:

    - By having a pre-configured data workflow, researchers can spend more time exploring different approaches on what to explore, describe and how to transform the data at hand, as well explore how to design the regression analysis methodology, without having to constantly modify the core structure of the project's framework. 


6. Any potential benefits or advantages to using the application:

    - The modular approach to the application improves the overall organization because it allows to produce and store different configurations that produce customized analysis reports in a quick efficient way. 


7. The cost associated with using the application:

    - It follows a pre-configured workflow, which doesn't allow to perform an analysis or statistical tests beyond what has been built-in. 

    - The implementation of new app features such as new statistical tests or machine learning support algorithms is more complex than the traditional approach.  

    - This version of the app specifically targets the problem of predicting medical insurance costs instead of being able to work with any data set or data science problem. One consequence of this is that some aspects of data transformation, are hard coded by default, e.g. running the app causes the variable 'region' to always be converted into a category type.   


8. The usability and user experience design of the application:

    - The use of the app is supposed to be simple as the user only has to change values in a set-up script which is pre-configured and run the app via Jupyter notebook.


10. The ongoing support and maintenance services available for the application

    - At this moment, there are no customized error/warning system if the quick set-up script is not well set (e.g. missing value, sub-section of the analysis activated when the parent section is deactivated, etc.), therefore, the user as to be knowledgeable to understand the logic of the set-up script to figure out what could be the problem. 


## To-do list:
    
    - Improve `data_transformation.py` functions so that these do not raise prompt errors if the arguments are changed. Same for `data_visualization.py`.
    - Expand the analysis to error distribution, error metrics comparison, etc:
        - Standardization of data.
        - Inclusion of features selection algorithms (forward/backward, etc).
        - Allow for a more complex randomization of data sets in order to allow for a k-fold cross validation process.

    - Consider removing the 'print results' functions into their own module instead of having a bloated 'modelling' script.
    - Also, consider moving some of the calculus made in `reg_results` (modeling.py) to statistical_testing.py if it makes sense.
    - Implement a mechanism that warns when the variables that will be used to start the regression section are not numerical.
    - Consider combining `x_var_container` and the `target_container` (parameters.py) into a dictionary or set a mechanism that warns when their length is not the same, to avoid mistakes in pairing features and target variable for regression purposes.


## Changes:

    - 'parameters.py' module implementation finished 
    - Created 'statistical_tests.py' to simplify the tasks in 'modelling.py'
    - Added two scatter plot functions to data_visualization.py - `resid_visual_analysis_1` and `resid_visual_analysis_2`, which belong exclusively to the residual analysis section.
    - Re-organized the residual analysis section so that the output is either related to train or to the test set, depending on the hyperparameter `residuals_set`.
    - Included the 'Compare Error Quality' section to the analysis workflow which summarizes the MSE/RMSE values of the stored regressions.
    - VIF analysis included to the workflow: turned it into a class so that multiple versions of the analysis can be performed and printed. 

    Additional changes:
    - Inclusion of the sections 1 and 2 of variable description in the exploratory data analysis of the workflow.
    - Inclusion of a 'query dataframe' section, activated by the hyperparameter `query_df`; this serves two purposes: i) allows to continue the analysis replacing the current dataframe by a queried version (queried versions can be produced and stored by instantiating the class `Filtered_DF` by setting the `queries` and `queried_df` hyperparameters.) ii) It allows to visually explore the relation of any explanatory variable with the target variable via scatter plot, by activating `scatter_query_df` and changing values in `feat_to_query`, `target_to_query` and `queries_scatter`.
    - Inclusion of a boxplot panel in the 'visualize' section.
    - Inclusion of a printed summary in the beginning of the Regression section, of the regressions to be performed.


## Issues / problems:

    - [still giving problems, needs revision] `nbconvert` (the app that converts the 'output.ipynb' file into a PDF documents) is connected to a browser, whose purpose it to use JavaScript code to process and render the HTML output of the notebooks. Because of that, when the cache of the browser is not cleared, nbconvert can sometimes convert older versions of the ipynb file instead of the current. The solution this problem is to clear the browser's cache. A better alternative would be to skip the Jupyter notebook as a means to export the PDF file and use a package such as `Reportlab`.


## Current task:

    - Inclusion of a Feature Selection section as a pre-regression part of the analysis.