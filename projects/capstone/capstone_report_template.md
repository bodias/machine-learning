# Machine Learning Engineer Nanodegree
## Capstone Project
Braian O. Dias
September 12th, 2018

## I. Definition

### Project Overview DONE
Financial data is growing exponentially, helping institutions to improve their relationships with customers, offering tailor made products and reducing the overall risk of a credit operation. Kaggle offers a great opportunity to make good use of machine learning techniques to address a real world problem in a financial institution which borrows money to people that are currently underserved with loans. The main goal of the Kaggle challenge named **Home Credit Default Risk** (https://www.kaggle.com/c/home-credit-default-risk) sponsored by Home Credit Group, is to make use of a variety of alternative data to predict their clients' repayment abilities.

This project will try to answer the main challenge question, *"Can you predict how capable each applicant is of repaying a loan?"* with a decent accuracy, taking into account the results of others challege's applicants. All the data needed to develop the solution is available on Kaggle in the form of .csv files that will be shown in detail later. As a current Fintech employee, which offers banking solutions to more than 700.000 customers in Brazil, it's a great opportunity to merge the Machine Learning techniques learned in the Nanodegree and apply it in my field of work.

### Problem Statement DONE
Home Credit is trying to minimize its loss due to loan defaults in a way that they accurately approve credit to customers that are likely to pay their debt. With supervised learning, we are able to build a model to predict their clients' repayment abilities, based on historical data provided by Home Credit through Kaggle.

The final solution will be built as follows :
1. Load the files provided by Kaggle;
2. Perform an Exploratory Analysis of the dataset;
3. Transform data to a suitable format to the machine learning algorithms
4. Train a baseline model, and a state of the art model
5. Evaluate the model results and adjust parameters
6. Predict the results on the test data and submit it to Kaggle to obtain the final score

The final score will be obtained after submitting the test predictions to Kaggle.

### Metrics DONE
The evaluation of the model will be done using **area under the ROC curve** (AUC) between the predicted probability and the observed target.
According to Google (https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc), "One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example", which means that it is a good choice for problems where the target variable is unbalanced. 
AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0. This is the metric defined by the Kaggle challenge to evaluate the best model.

## II. Analysis

### Data Exploration DONE
The data provided by Kaggle is comprised of 8 CSV files, with a main train/test file with reference to all the other files through the SK_xxx columns. The file *"HomeCredit_columns_description.csv"* contains information about each column in each file.
Below is a summary of all 8 files available : 1 main file for training (with target) 1 main file for testing (without the target), and 6 other files containing additional information about each loan.

![File stats](home_credit/images/file_stats.png)

The training data has 307511 observations (each one a separate loan) and 122 features (variables) including the TARGET (the label we want to predict). The test data folows the same structure, but it has 48744 observations and lacks the TARGET column.

We can see the first 5 observations of the main training data below :

![training data overview](home_credit/images/app_train_head.png)

There are many features in the dataset, so in order to analyse them using the proper way the dataset will be splitted in numerical and non-numerical values. Then numerical features will be splitted in integer and float.

* There are 16 NON-numerical features in the main dataset.
* There are 104 numerical features in the main dataset.
    * 39 are Int64 features
    * 65 are Float64 features

#### Numerical features
First, we will compute some basic statistics to determine how values are distributed and to try to infer the purpose of each one, and also to detect anomalies.
Below are the statistics of the **integer features**, calculated through the *describe()* method of the pandas dataframe : 

![integer features statistics](home_credit/images/int_features_stats.png)

It can be seen that most of the integer features are in fact *binary features* already coded in [0,1], then those features will be classified as *int_binary_features* for the sake of feature transformation that will be performed later.
Also, there are features that represent count of days : ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH']. One of them, ***DAYS_EMPLOYED***, which stores the count of days the applicant is employed, has a maximum value of 365243, which is more 1000 years. Let's investigate further and see how many observations have this pattern using a histogram.

![Days of employment histogram](home_credit/images/DAYS_EMPLOYMENT_HIST.png)

A total of **55374 days of employment** have issues. For now, nothing will be done about this anomaly, but it will be treated in the section *Data Preprocessing* later on this document.
Now, let's focus on the remaining numerical values, the **float features**. As previously, some basic statistics about the data was generated usin the *describe()* method :

![float features statistics](home_credit/images/float_features_stats.png)

Now we start to see some differences between the count of each feature and the total count of observations in the dataset, which means there is **missing data** in these features. There is a total of 61 out of 65 float features with missing data, and the first 10 of them are presented below with the percentage of null values over the whole dataset.

![float features - null values](home_credit/images/float_features_null.png)

The table above shows that these top 10 features by missing count have about 68% of missing values over the whole dataset. This is a huge number, and the consequences of this characteristic of the data will be discussed later on the *Data Preprocessing* section.

#### Non-numerical features
The remaining class of features holds discrete data coded as text. Again python can output basic statistics of this data type, but instead of min, max, quantiles, mean and standard deviation, the method *describe()* give us an overview of the frequency and unique values of the discrete attribute. Below is the output of the pandas describe() method over the discrete data : 

![non-numerical features statistics](home_credit/images/non_numerical_features_stats.png)

We can see from the data above that the features EMERGENCYSTATE_MODE has only two possible values (yes/no). However, this features is going to be treated as it has more than 2 categories, once we're going to introduce a new category to represent the NaN values. Other 3 features, ['NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY'], have binary values, then they will be treated as *text_binary_features* in the data preprocessing phase .
It's also possible to note that there are **missing** values for some features, to understand this issue better, the following table presents the percentage of null values over the whole dataset.

![non-numerical features - null values](home_credit/images/non_numerical_features_null.png)

6 out of 16 discrete features has missing values, and 3 of them have more than half of the values missing. The strategies for data imputation will be discussed on the *Data Preprocessing* section.

To summarize, the whole main training dataset was analyzed accordingly to each feature data type. This process also separated the features into continuous (int and float features) and discrete (int/text binary features, discrete with many levels) features. With this distinction it will be possible to perform *label encoding* over discrete features in order to have them in a suitable format to train a machine learning model.

### Exploratory Visualization

Looking at the training data, it's possible to note that the target variable is not balanced:

* Number of training instances with TARGET 0 : 282686
* Number of training instances with TARGET 1 : 24825

![Project Design flow](home_credit/images/target_var_dist.png)



In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

### Benchmark
In this section, you will need to provide a clearly defined benchmark result or threshold for comparing across performances obtained by your solution. The reasoning behind the benchmark (in the case where it is not an established result) should be discussed. Questions to ask yourself when writing this section:
- _Has some result or value been provided that acts as a benchmark for measuring performance?_
- _Is it clear how this result or value was obtained (whether by data or by hypothesis)?_


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
