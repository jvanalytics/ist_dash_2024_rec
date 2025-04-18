import pandas as pd
import numpy as np

def fill_nulls_based_on_top_value_multiple_columns(df, target_column, groupby_columns):
    """
    Fills null values in the target column based on the most frequent values for each group of one or more columns.

    Args:
        df: The pandas DataFrame.
        target_column: The column with null values to be filled.
        groupby_columns: The column(s) used to group and determine the most frequent value (can be a string or a list).

    Returns:
        A pandas DataFrame with the null values in the target column filled.
    """
    # Define a function to get the most frequent value in each group
    def most_frequent(x):
        # Return the most frequent value, or None if there are no values
        return x.mode()[0] if not x.mode().empty else None

    # Create a Series to hold the fill values for each group
    fill_values = df.groupby(groupby_columns)[target_column].transform(most_frequent)

    # Fill the target column with the computed fill values
    df[target_column] = df[target_column].fillna(fill_values)

    return df


def encode_and_sort_by_value_counts(data, column_name, ascending=True):
    """
    Encodes categorical values in a specified column and sorts the resulting category mapping by value counts.

    Args:
        data: The pandas DataFrame.
        column_name: The name of the column to encode and sort.
        ascending: Whether to sort in ascending order (default is descending).

    Returns:
        A category mapping sorted by value counts in the specified order.
    """
    
    # Calculate the value counts and sort them by the specified order
    value_counts = data[column_name].value_counts(ascending=ascending)
    
    # Create a mapping dictionary, now based on sorted value counts
    category_mapping = {category: index + 1 for index, category in enumerate(value_counts.index)}
    print(category_mapping)
       
    return category_mapping



def encode_and_sort_alphabetically_values(data, column_name, ascending=True):
    """
    Encodes categorical values in a specified column and sorts the resulting encoded values alphabetically.

    Args:
        data: The pandas DataFrame.
        column_name: The name of the column to encode and sort.
        ascending: Whether to sort in ascending or descending alphabetical order.

    Returns:
        A category mapping of all values in the column.
    """
    
    # Get unique categories and sort them alphabetically
    unique_categories = sorted(data[column_name].unique(), reverse=not ascending)
    
    # Create a mapping dictionary, now based on sorted unique categories
    category_mapping = {category: index + 1 for index, category in enumerate(unique_categories)}
       
    print(category_mapping)
    
    return category_mapping



def encode_and_sort_by_value_counts(data, column_name, ascending=True):
    """
    Encodes categorical values in a specified column and sorts the resulting category mapping by value counts.

    Args:
        data: The pandas DataFrame.
        column_name: The name of the column to encode and sort.
        ascending: Whether to sort in ascending order (default is descending).

    Returns:
        A category mapping sorted by value counts in the specified order.
    """
    
    # Calculate the value counts and sort them by the specified order
    value_counts = data[column_name].value_counts(ascending=ascending)
    
    # Create a mapping dictionary, now based on sorted value counts
    category_mapping = {category: index + 1 for index, category in enumerate(value_counts.index)}
    print(category_mapping)
       
    return category_mapping


def calculate_sessions_per_user_ratio(df, groupby_columns):
    """
    Calculates the number of unique users, number of sessions, and session-to-user ratio per group.

    Args:
        df: The pandas DataFrame.
        groupby_columns: A list of column names to group by.

    Returns:
        A pandas DataFrame with the number of users, sessions, and sessions-per-user for each group.
    """
    # Ensure that groupby_columns is a list (if only one column is passed as a string)
    if isinstance(groupby_columns, str):
        groupby_columns = [groupby_columns]
    
    # Aggregate to calculate unique users and sessions per group
    agg_table = df.groupby(groupby_columns).agg(
        users=('user_pseudo_id', 'nunique'),          # Count of unique users
        sessions=('session_id', 'nunique')            # Count of unique sessions
    ).reset_index()

    # Calculate the session-to-user ratio
    agg_table['sessions_per_user'] = agg_table['sessions'] / agg_table['users']
    
    
    # Sort by sessions_per_user in descending order
    agg_table.sort_values(by='sessions', ascending=False, inplace=True)
    
    return agg_table



def calculate_returning_user_percentage_by_group(df, groupby_column):
    """
    Calculates the total count of events and the percentage of returning users
    for a specified group by column based on event timestamps. 
    If 'returning_user' does not exist, it will be created.

    Args:
        df: The pandas DataFrame.
        groupby_column: The column name to group by.

    Returns:
        A pandas DataFrame with the total count of events and percentage of returning users for each group.
    """
    # Ensure that the groupby_column is a list (if only one column is passed as a string)
    if isinstance(groupby_column, str):
        groupby_column = [groupby_column]

    # Check if 'returning_user' column exists
    if 'returning_user' not in df.columns:
        # Create the returning_user column based on ga_session_number
        df['returning_user'] = df['ga_session_number'].apply(lambda x: 0 if x == 1 else 1)

    # Aggregate by the user-specified group column
    agg_table = df.groupby(groupby_column).agg(
        total_events=('event_timestamp', 'count'),                      # Count total events
        returning_users=('returning_user', lambda x: (x == 1).sum())  # Count returning users
    ).reset_index()

    # Calculate the percentage of returning users
    agg_table['returning_user_percentage'] = (agg_table['returning_users'] / agg_table['total_events']) * 100
    
    # Sort by returning_user_percentage in descending order
    agg_table.sort_values(by='returning_user_percentage', ascending=False, inplace=True)
    
    return agg_table

from math import pi, sin, cos


# Generalized function to encode any column based on a mapping dictionary
def encode_column_with_mapping(df, column, mapping_dict, default_value=np.nan):
    """
    Encode a column based on a mapping dictionary.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the column to encode.
    column (str): The column name to encode.
    mapping_dict (dict): Dictionary where keys are the original values and values are the encodings.
    default_value: Value to assign if the key is not found in the mapping dict (default is NaN).
    
    Returns:
    pd.Series: Encoded column.
    """
    return df[column].map(mapping_dict).fillna(default_value)



def sheet_to_dict(excel_file, sheet_name):
    """
    Converts a sheet in an Excel file to a dictionary with the first column as the key and the second as the value.
    
    Args:
    excel_file (str): Path to the Excel file.
    sheet_name (str): Name of the sheet to convert.
    
    Returns:
    dict: A dictionary with keys from the first column and values from the second column.
    """
    # Read the sheet into a DataFrame
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # Ensure that there are at least two columns
    if df.shape[1] < 2:
        raise ValueError(f"The sheet '{sheet_name}' does not have at least two columns.")
    
    # Create a dictionary from the first and second columns
    mapping_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    
    return mapping_dict




from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig, show
from dslabs_functions import get_variable_types, plot_bar_chart, HEIGHT
from pandas import DataFrame


    
def analyse_property_granularity(
    data: DataFrame, property: str, vars: list[str], file_tag=''
) -> ndarray:
    cols: int = len(vars)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"{file_tag} Granularity study for {property}")
    for i in range(cols):
        counts: Series[int] = data[vars[i]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[0, i],
            title=vars[i],
            xlabel=vars[i],
            ylabel="nr records",
            percentage=False,
        )
    return axs


from numpy import array, ndarray
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_bar_chart




def naive_Bayes_study(
    trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy", file_tag=''
) -> tuple:
    estimators: dict = {
        "GaussianNB": GaussianNB(),
        # "MultinomialNB": MultinomialNB(),
        "BernoulliNB": BernoulliNB(),
    }

    xvalues: list = []
    yvalues: list = []
    best_model = None
    best_params: dict = {"name": "", "metric": metric, "params": ()}
    best_performance = 0
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY: array = estimators[clf].predict(tstX)
        eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
        if eval - best_performance > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["name"] = clf
            best_params[metric] = eval
            best_model = estimators[clf]
        yvalues.append(eval)
        # print(f'NB {clf}')
    plot_bar_chart(
        xvalues,
        yvalues,
        title=f"{file_tag} Naive Bayes Models ({metric})",
        ylabel=metric,
        percentage=True,
    )

    return best_model, best_params



from itertools import product
from numpy import ndarray, set_printoptions, arange
from matplotlib.pyplot import gca, cm
from matplotlib.axes import Axes

def plot_confusion_matrix(cnf_matrix: ndarray, classes_names: ndarray, ax: Axes = None) -> Axes:  # type: ignore
    if ax is None:
        ax = gca()
    title = "Confusion matrix"
    set_printoptions(precision=2)
    tick_marks: ndarray = arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cnf_matrix, interpolation="nearest", cmap=cm.Blues)

    for i, j in product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        ax.text(
            j, i, format(cnf_matrix[i, j], "d"), color="y", horizontalalignment="center"
        )
    return ax



def mvi_by_filling(data: DataFrame, strategy: str = "frequent") -> DataFrame:
    df: DataFrame
    variables: dict = get_variable_types(data)
    stg_num, v_num = "mean", -1
    stg_sym, v_sym = "most_frequent", "NA"
    stg_bool, v_bool = "most_frequent", False
    if strategy != "knn":
        lst_dfs: list = []
        if strategy == "constant":
            stg_num, stg_sym, stg_bool = "constant", "constant", "constant"
        if len(variables["numeric"]) > 0:
            imp = SimpleImputer(strategy=stg_num, fill_value=v_num, copy=True)
            tmp_nr = DataFrame(
                imp.fit_transform(data[variables["numeric"]]),
                columns=variables["numeric"],
            )
            lst_dfs.append(tmp_nr)
        if len(variables["symbolic"]) > 0:
            imp = SimpleImputer(strategy=stg_sym, fill_value=v_sym, copy=True)
            tmp_sb = DataFrame(
                imp.fit_transform(data[variables["symbolic"]]),
                columns=variables["symbolic"],
            )
            lst_dfs.append(tmp_sb)
        if len(variables["binary"]) > 0:
            imp = SimpleImputer(strategy=stg_bool, fill_value=v_bool, copy=True)
            tmp_bool = DataFrame(
                imp.fit_transform(data[variables["binary"]]),
                columns=variables["binary"],
            )
            lst_dfs.append(tmp_bool)
        df = concat(lst_dfs, axis=1)
    else:
        imp = KNNImputer(n_neighbors=10)
        imp.fit(data)
        ar: ndarray = imp.transform(data)
        df = DataFrame(ar, columns=data.columns, index=data.index)
    return df



def remove_columns_with_na(df):
    df_copy = df.copy()
    df_cols = df_copy.columns

    # Identify columns with any missing values
    cols_with_na = [col for col in df_cols if df_copy[col].isna().any()]

    # Print the list of columns that have missing values
    print(f"Columns with missing values: {cols_with_na}")

    # Remove columns that have any missing values
    df_copy = df_copy.drop(columns=cols_with_na)

    return df_copy



from typing import Literal
from numpy import array, ndarray
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart
from dslabs_functions import read_train_test_from_files, plot_evaluation_results

def knn_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, k_max: int=19, lag: int=2, metric='accuracy', file_tag=''
        ) -> tuple[KNeighborsClassifier | None, dict]:
    dist: list[Literal['manhattan', 'euclidean', 'chebyshev']] = ['manhattan', 'euclidean', 'chebyshev']

    kvalues: list[int] = [i for i in range(1, k_max+1, lag)]
    best_model: KNeighborsClassifier | None = None
    best_params: dict = {'name': 'KNN', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict[str, list] = {}
    for d in dist:
        y_tst_values: list = []
        for k in kvalues:
            clf = KNeighborsClassifier(n_neighbors=k, metric=d)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance: float = eval
                best_params['params'] = (k, d)
                best_model = clf
            # print(f'KNN {d} k={k}')
        values[d] = y_tst_values
    print(f'KNN best with k={best_params['params'][0]} and {best_params['params'][1]}')
    plot_multiline_chart(kvalues, values, title=f'{file_tag} KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)

    return best_model, best_params


# Define a function to sample 10% from each group
def sample_per_day(group, fraction=0.1):
    return group.sample(frac=fraction)


import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure, show

def define_target_and_prepare_data(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, pd.Series, List[int], Dict[str, List[int]]]:
    """
    Defines the target variable and prepares the feature set and target labels.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data.
    target (str): The name of the target variable.

    Returns:
    Tuple[np.ndarray, pd.Series, List[int], Dict[str, List[int]]]: A tuple containing:
        - y (pd.Series): The target variable.
        - X (np.ndarray): The feature set.
        - labels (List[int]): Sorted list of unique labels.
        - values (Dict[str, List[int]]): Counts of original classes.
    
    Raises:
    KeyError: If the target variable is not found in the DataFrame.
    """
    
    # Check if the target exists in the DataFrame
    if target not in df.columns:
        raise KeyError(f"The target column '{target}' does not exist in the DataFrame. Available columns are: {df.columns.tolist()}")

    # Extract labels and sort them
    labels: list = list(df[target].unique())
    labels.sort()
    print(f"Labels={labels}")

    # Create a dictionary to store original class counts
    values: dict[str, list[int]] = {
        "Original": [
            len(df[df[target] == 0]),  # Assuming 0 is the negative class
            len(df[df[target] == 1]),  # Assuming 1 is the positive class
        ]
    }

    y: pd.Series = df.pop(target)  # Keep y as a Series
    X: np.ndarray = df.values  # Extract the features as ndarray

    return y, X, labels, values


def split_data_save_csv(X: pd.DataFrame, y: pd.Series, data_columns: List[str], target_column: str, file_tag=None, train_size=0.7, save=False, save_path="data/"):
    """
    Splits data into training and test sets, then returns the corresponding DataFrames.
    Optionally saves the DataFrames as CSV files.
    
    Parameters:
    X (pd.DataFrame): The feature set.
    y (pd.Series): The target labels.
    data_columns (list): The column names of X.
    target_column (str): The name of the target column.
    file_tag (str, optional): Tag to use in the filename if saving CSVs.
    train_size (float, optional): Proportion of data to use for training (default is 0.7).
    save (bool, optional): Whether to save the train/test DataFrames as CSV files (default is False).
    save_path (str, optional): Path to save the CSV files (default is "data/").    
    Returns:
    train (pd.DataFrame): The training DataFrame (features + target).
    test (pd.DataFrame): The testing DataFrame (features + target).
    """
    # Calculate the split index based on train_size
    split_index = int(len(X) * train_size)

    # Split the DataFrame into train and test sets
    train_X = X.iloc[:split_index]
    test_X = X.iloc[split_index:]

    # Separate the target variable
    train_y = y.iloc[:split_index]
    test_y = y.iloc[split_index:]

    # Create train and test DataFrames including the target
    train = pd.concat([train_X.reset_index(drop=True), train_y.reset_index(drop=True)], axis=1)
    test = pd.concat([test_X.reset_index(drop=True), test_y.reset_index(drop=True)], axis=1)

    # Optionally save to CSV
    if save and file_tag:
        train.to_csv(f"{save_path}{file_tag}_train.csv", index=False)
        test.to_csv(f"{save_path}{file_tag}_test.csv", index=False)

    return train, test


    
from typing import Tuple, Optional

def simple_split_df(df: pd.DataFrame, sort_by: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train and test sets based on a 70/30 split ratio.
    Optionally, sorts the DataFrame by a specified column before splitting.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be split.

    sort_by : Optional[str], default=None
        The column name by which to sort the DataFrame before splitting.
        If None, no sorting will be applied.

    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the training set (70%) and test set (30%).

    Example:
    --------
    train, test = simple_split_df(data_save, sort_by="date")
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Sort the DataFrame by the specified column if provided
    if sort_by:
        df_copy.sort_values(by=sort_by, inplace=True)

    # Determine the split index (70% train, 30% test)
    split_index = int(len(df_copy) * 0.7)

    # Split the DataFrame into train and test sets
    train = df_copy.iloc[:split_index]
    test = df_copy.iloc[split_index:]

    return train, test




from numpy import ndarray
from pandas import DataFrame, read_csv
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN


def evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "returning_user", metric: str = "accuracy"
) -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    return eval


from typing import Literal
from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig, show
from sklearn.tree import DecisionTreeClassifier
from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, read_train_test_from_files
from dslabs_functions import plot_evaluation_results, plot_multiline_chart


def trees_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, d_max: int=10, lag:int=2, metric='accuracy', file_tag=''    
        ) -> tuple:
    criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
    depths: list[int] = [i for i in range(2, d_max+1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict = {}
    for c in criteria:
        y_tst_values: list[float] = []
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['params'] = (c, d)
                best_model = clf
            # print(f'DT {c} and d={d}')
        values[c] = y_tst_values
    print(f' DT best with {best_params['params'][0]} and d={best_params['params'][1]}')
    plot_multiline_chart(depths, values, title=f'{file_tag} DT Models ({metric})', xlabel='d', ylabel=metric, percentage=True)

    return best_model, best_params

from numpy import array, ndarray
from matplotlib.pyplot import figure, savefig, show
from sklearn.linear_model import LogisticRegression
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs_functions import plot_evaluation_results, plot_multiline_chart



def logistic_regression_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
    file_tag=''    
    
) -> tuple[LogisticRegression | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    penalty_types: list[str] = ["l1", "l2"]  # only available if optimizer='liblinear'

    best_model = None
    best_params: dict = {"name": "LR", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    for type in penalty_types:
        warm_start = False
        y_tst_values: list[float] = []
        for j in range(len(nr_iterations)):
            clf = LogisticRegression(
                penalty=type,
                max_iter=lag,
                warm_start=warm_start,
                solver="liblinear",
                verbose=False,
            )
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            warm_start = True
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params["params"] = (type, nr_iterations[j])
                best_model: LogisticRegression = clf
            # print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
        values[type] = y_tst_values
    plot_multiline_chart(
        nr_iterations,
        values,
        title=f"LR models ({metric})",
        xlabel="nr iterations",
        ylabel=metric,
        percentage=True,
    )
    print(
        f'LR best for {best_params["params"][1]} iterations (penalty={best_params["params"][0]})'
    )

    return best_model, best_params


from typing import Literal
from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, show,suptitle
from sklearn.neural_network import MLPClassifier
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart


def mlp_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_iterations: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005],
    
    file_tag=''    
) -> tuple[MLPClassifier | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
        "constant",
        "invscaling",
        "adaptive",
    ]  # only used if optimizer='sgd'
    # learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    _, axs = subplots(
        1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
    )
    
    suptitle(f"{file_tag} MLP study for {metric}")
    
    for i in range(len(lr_types)):
        type: str = lr_types[i]
        values = {}
        for lr in learning_rates:
            warm_start: bool = False
            y_tst_values: list[float] = []
            for j in range(len(nr_iterations)):
                clf = MLPClassifier(
                    learning_rate=type,
                    learning_rate_init=lr,
                    max_iter=lag,
                    warm_start=warm_start,
                    activation="logistic",
                    solver="sgd",
                    verbose=False,
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                warm_start = True
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (type, lr, nr_iterations[j])
                    best_model = clf
                print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            nr_iterations,
            values,
            ax=axs[0, i],
            title=f"MLP with {type}",
            xlabel="nr iterations",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'MLP best for {best_params["params"][2]} iterations (lr_type={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params


from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.ensemble import RandomForestClassifier
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)

from dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart


def random_forests_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
    file_tag=''
) -> tuple[RandomForestClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    max_features: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: RandomForestClassifier | None = None
    best_params: dict = {"name": "RF", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}

    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)

    suptitle(f"{file_tag} Random Forests study for {metric}")
    
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for f in max_features:
            y_tst_values: list[float] = []
            for n in n_estimators:
                clf = RandomForestClassifier(
                    n_estimators=n, max_depth=d, max_features=f
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (d, f, n)
                    best_model = clf
                # print(f'RF d={d} f={f} n={n}')
            values[f] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"RF max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'RF best for {best_params["params"][2]} trees (d={best_params["params"][0]} and f={best_params["params"][1]})'
    )
    return best_model, best_params


from numpy import array, ndarray
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.ensemble import GradientBoostingClassifier
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart


def gradient_boosting_study(
    trnX: ndarray,
    trnY: array,
    tstX: ndarray,
    tstY: array,
    nr_max_trees: int = 2500,
    lag: int = 500,
    metric: str = "accuracy",
    file_tag=''
) -> tuple[GradientBoostingClassifier | None, dict]:
    n_estimators: list[int] = [100] + [i for i in range(500, nr_max_trees + 1, lag)]
    max_depths: list[int] = [2, 5, 7]
    learning_rates: list[float] = [0.1, 0.3, 0.5, 0.7, 0.9]

    best_model: GradientBoostingClassifier | None = None
    best_params: dict = {"name": "GB", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    cols: int = len(max_depths)
    _, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    
    suptitle(f"{file_tag} Gradient Boosting study for {metric}")
    
    for i in range(len(max_depths)):
        d: int = max_depths[i]
        values = {}
        for lr in learning_rates:
            y_tst_values: list[float] = []
            for n in n_estimators:
                clf = GradientBoostingClassifier(
                    n_estimators=n, max_depth=d, learning_rate=lr
                )
                clf.fit(trnX, trnY)
                prdY: array = clf.predict(tstX)
                eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
                y_tst_values.append(eval)
                if eval - best_performance > DELTA_IMPROVE:
                    best_performance = eval
                    best_params["params"] = (d, lr, n)
                    best_model = clf
                # print(f'GB d={d} lr={lr} n={n}')
            values[lr] = y_tst_values
        plot_multiline_chart(
            n_estimators,
            values,
            ax=axs[0, i],
            title=f"GB max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'GB best for {best_params["params"][2]} trees (d={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params

from numpy import ndarray
from pandas import concat
from sklearn.impute import SimpleImputer, KNNImputer
from dslabs_functions import get_variable_types, mvi_by_filling


def apply_missing_values_frequent(df):

    df_copy=df.copy()
    df_copy = mvi_by_filling(df_copy, strategy="frequent")

    return df_copy


def apply_missing_values_remove_cols_and_any_na_rows(df,cols):

    df_copy=df.copy()

    # pass cols list to drop
    df_copy = df_copy.drop(cols, axis=1)

    # drop remaining records where there are nulls
    df_copy = df_copy.dropna(axis=0, how="any")

    return df_copy


from dslabs_functions import (
    NR_STDEV,
    get_variable_types,
    determine_outlier_thresholds_for_var,
)

def truncate_outliers(df,summary5_df,outlier_var):

    df_copy=df.copy()
    
    # this script is available in data_functions originally from DSLabs site in Outlier chapter
    top, bottom = determine_outlier_thresholds_for_var(summary5_df[outlier_var])
    df_copy[outlier_var] = df_copy[outlier_var].apply(
        lambda x: top if x > top else bottom if x < bottom else x
    )

    print("Data after truncating outliers:", df_copy.shape)


    return df_copy


from pandas import DataFrame, Series

def drop_outliers(df,summary5_df,outlier_var):

    df_copy=df.copy()
    
    # this script is available in data_functions originally from DSLabs site in Outlier chapter
    top, bottom = determine_outlier_thresholds_for_var(summary5_df[outlier_var])
    outliers: Series = df_copy[(df_copy[outlier_var] > top) | (df_copy[outlier_var] < bottom)]

    df_copy.drop(outliers.index, axis=0, inplace=True)

    print("Data after truncating outliers:", df_copy.shape)

    return df_copy


from sklearn.preprocessing import StandardScaler

def apply_standard_scaler(df: DataFrame) -> DataFrame:
    """
    Apply StandardScaler to the entire DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to scale.

    Returns:
    pd.DataFrame: The scaled DataFrame.
    """
    df_copy = df.copy()
    
    # Apply scaling to the entire DataFrame
    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_copy)
    df_zscore = DataFrame(transf.transform(df_copy), index=df_copy.index, columns=df_copy.columns)

    return df_zscore


from sklearn.preprocessing import MinMaxScaler

def apply_min_max_scaler(df: DataFrame) -> DataFrame:
    df_copy = df.copy()
    
    # Apply MinMax scaling to the entire DataFrame including the target column
    transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_copy)
    df_minmax = DataFrame(transf.transform(df_copy), index=df_copy.index, columns=df_copy.columns)

    return df_minmax


def apply_remove_low_variance_variables(df: DataFrame,max_threshold=0.024,min_features_to_keep=10,exclude=['day_of_year'],target='returning_user') -> DataFrame:

    from dslabs_functions import select_low_variance_variables
    # this script is available in data_functions originally from DSLabs site in Feature Selection chapter

    df_copy=df.copy()
    
    vars2drop: list[str] = select_low_variance_variables(df_copy, max_threshold=max_threshold, min_features_to_keep=min_features_to_keep,exclude=exclude, target=target)
    
    print("columns to drop:", vars2drop)

    df_vars_drop = df_copy.drop(columns=vars2drop, errors='ignore')
    
    print("Remaining columns:", df_vars_drop.columns)

    return df_vars_drop


def apply_remove_redundant_variables(df: DataFrame,min_threshold=0.4,exclude=['day_of_year'], target='returning_user')-> DataFrame:

    from dslabs_functions import select_redundant_variables
    # this script is available in data_functions originally from DSLabs site in Feature Selection chapter
    
    df_copy=df.copy()

    vars2drop = select_redundant_variables(df_copy, min_threshold=min_threshold,exclude=exclude, target=target)
    print("columns to drop:", vars2drop)

    df_vars_drop = df_copy.drop(columns=vars2drop, errors='ignore')
    
    print("Remaining columns:", df_vars_drop.columns)


    return df_vars_drop


def apply_upsample_negative_class(df: pd.DataFrame, target: str = 'returning_user', desired_ratio: float = 0.85, sort_by: str = 'day_of_year') -> pd.DataFrame:
    df_copy = df.copy()

    positive_class = 1  # Assuming positive class is 1
    negative_class = 0  # Assuming negative class is 0

    # Separate positive and negative classes
    df_positive = df_copy[df_copy[target] == positive_class]  # Positive class
    df_negative = df_copy[df_copy[target] == negative_class]  # Negative class

    # Calculate total desired size based on keeping positive class fixed
    current_positive_size = len(df_positive)
    total_desired_size = int(current_positive_size / (1 - desired_ratio))

    # Calculate how many negatives we need to reach the desired total size
    target_negative_size = total_desired_size - current_positive_size

    # Upsample the negative class to match the target size
    df_negative_upsampled = df_negative.sample(n=target_negative_size, replace=True, random_state=42)

    # Combine positive class with the upsampled negative class
    df_balanced = pd.concat([df_positive, df_negative_upsampled])

    # Reset the index and sort the dataset
    df_balanced.reset_index(drop=True, inplace=True)
    df_balanced.sort_values(by=sort_by, inplace=True)

    # Print new class distribution
    print(f"Balanced class distribution:\n{df_balanced[target].value_counts(normalize=True) * 100}\n")
    print(df_balanced.shape)
    return df_balanced


def apply_balanced_downsampling(df: DataFrame,target='returning_user',sort_by='day_of_year') -> DataFrame:

    df_copy=df.copy()


    # Ensure positive_class and negative_class are defined and match the target values
    positive_class = 1  # Or whatever your positive class value is
    negative_class = 0  # Or whatever your negative class value is

    # Separate the majority and minority classes
    df_majority = df_copy[df_copy[target] == negative_class]
    df_minority = df_copy[df_copy[target] == positive_class]

    # Check the class distribution
    print(f"Original class distribution:\n{df_copy[target].value_counts(normalize=True) * 100}\n")

    # Downsample the majority class to match the size of the minority class
    df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

    # Combine the downsampled majority class with the minority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])

    # sort  the combined dataset
    df_balanced.sort_values(by=sort_by, inplace=True)
    

    # Check the new class distribution to verify the balance
    print(f"Balanced class distribution:\n{df_balanced[target].value_counts(normalize=True) * 100}\n")

    
    return df_balanced


def apply_balanced_hybrid(df, target='returning_user', minority_ratio=0.5,sort_by='day_of_year'):
    # Create a copy of the dataframe
    df_copy = df.copy()

    # Define positive and negative classes
    positive_class = 1  # Modify as per your positive class value
    negative_class = 0  # Modify as per your negative class value

    # Separate the majority and minority classes
    df_majority = df_copy[df_copy[target] == negative_class]
    df_minority = df_copy[df_copy[target] == positive_class]

    # Check the current class distribution
    print(f"Original class distribution:\n{df_copy[target].value_counts(normalize=True) * 100}\n")

    # Sort by 'day_of_year' (or another time-related feature) to ensure the data is split based on time
    df_majority.sort_values(by=sort_by, inplace=True)
    df_minority.sort_values(by=sort_by, inplace=True)

    # Determine the desired size for the final dataset
    total_majority_samples = len(df_majority)
    # total_minority_samples = len(df_minority)
    
    # Set the ratio for the majority and minority classes
    desired_minority_ratio = minority_ratio
    desired_majority_ratio = 1-minority_ratio


    # Calculate the new majority size (XX% of the total number of majority samples)
    downsampled_majority_size = int(desired_majority_ratio * total_majority_samples)

    # Calculate the corresponding upsampled minority size (XX% of the downsampled majority size)
    upsampled_minority_size = int(downsampled_majority_size * (desired_minority_ratio / desired_majority_ratio))
    
    # Downsample the majority class (keeping the older data based on 'day_of_year')
    df_majority_downsampled = df_majority.head(downsampled_majority_size)

    # Upsample the minority class to match the desired minority size for a XX/XX split
    df_minority_upsampled = df_minority.sample(n=upsampled_minority_size, replace=True)

    # Combine the downsampled majority and upsampled minority classes
    df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])

    # Sort the dataset by 'day_of_year' again if needed
    df_balanced.sort_values(by=sort_by, inplace=True)

    # Check the new class distribution
    print(f"Hybrid class distribution ({desired_majority_ratio*100}/{desired_minority_ratio*100}):\n{df_balanced[target].value_counts(normalize=True) * 100}\n")

    return df_balanced



def apply_balanced_smote(df,target='returning_user',sort_by='day_of_year'):

    from imblearn.over_sampling import SMOTE

    # Create a copy of the dataframe
    df_copy = df.copy()

    print(f"Original class distribution:\n{df_copy[target].value_counts(normalize=True) * 100}\n")


    # Separate the features (X) and the target (y)
    X = df_copy.drop(columns=[target])
    y = df_copy[target]

    # Apply SMOTE to balance the data
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Recombine the features and target into a single dataframe
    df_smote = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=[target])], axis=1)

    # Sort the dataset by 'day_of_year' again if needed
    df_smote.sort_values(by=sort_by, inplace=True)

    print(f"New class distribution:\n{df_smote[target].value_counts(normalize=True) * 100}\n")


    return df_smote


import os
# from openpyxl import load_workbook


# # Function to append distinct combinations of selected columns into sheets in an Excel file
# def append_columns_to_excel(df, columns_dict, output_file):
#     """
#     Append distinct combinations of selected columns into separate sheets in an existing Excel file,
#     with the columns ordered by their names for easier hierarchical encoding and add an empty encoding column.

#     Args:
#     df (pd.DataFrame): The DataFrame containing the columns to save.
#     columns_dict (dict): Dictionary where keys are sheet names, and values are lists of column names to include.
#     output_file (str): The path of the Excel file to save the sheets.

#     Returns:
#     None
#     """
#     # Check if the file exists and is a valid Excel file
#     if os.path.exists(output_file):
#         try:
#             # Try to load the existing workbook
#             with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
#                 writer.book = load_workbook(output_file)
                
#                 # Loop over each sheet name and corresponding list of columns
#                 for sheet_name, columns in columns_dict.items():
#                     # Check if all the specified columns exist in the DataFrame
#                     missing_columns = [col for col in columns if col not in df.columns]
#                     if missing_columns:
#                         print(f"Warning: The following columns are not found in the DataFrame for sheet '{sheet_name}': {missing_columns}")
#                         continue

#                     # Get distinct combinations of the selected columns
#                     distinct_values = df[columns].drop_duplicates().dropna(how='all')

#                     # Convert columns to strings temporarily for sorting to avoid float-string comparison errors
#                     distinct_values = distinct_values.astype(str)

#                     # Sort distinct values by the specified columns for hierarchical grouping
#                     distinct_values.sort_values(by=columns, inplace=True)

#                     # Add an empty encoding column for each column in the DataFrame
#                     for col in columns:
#                         distinct_values[f'{col}_enc'] = pd.NA

#                     # Write distinct values to a new sheet named after the sheet_name
#                     distinct_values.to_excel(writer, sheet_name=sheet_name, index=False)
                    
#         except Exception as e:
#             print(f"Error: {e}")
#             print("The file might be corrupt or invalid. Creating a new file.")
#             # Create a new file if loading fails
#             with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
#                 for sheet_name, columns in columns_dict.items():
#                     missing_columns = [col for col in columns if col not in df.columns]
#                     if missing_columns:
#                         print(f"Warning: The following columns are not found in the DataFrame for sheet '{sheet_name}': {missing_columns}")
#                         continue

#                     # Get distinct combinations of the selected columns
#                     distinct_values = df[columns].drop_duplicates().dropna(how='all')

#                     # Convert columns to strings temporarily for sorting
#                     distinct_values = distinct_values.astype(str)

#                     # Sort distinct values by the specified columns for hierarchical grouping
#                     distinct_values.sort_values(by=columns, inplace=True)

#                     # Add an empty encoding column for each column in the DataFrame
#                     for col in columns:
#                         distinct_values[f'{col}_enc'] = pd.NA

#                     distinct_values.to_excel(writer, sheet_name=sheet_name, index=False)

#     else:
#         # If the file does not exist, create a new one
#         with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
#             for sheet_name, columns in columns_dict.items():
#                 missing_columns = [col for col in columns if col not in df.columns]
#                 if missing_columns:
#                     print(f"Warning: The following columns are not found in the DataFrame for sheet '{sheet_name}': {missing_columns}")
#                     continue

#                 # Get distinct combinations of the selected columns
#                 distinct_values = df[columns].drop_duplicates().dropna(how='all')

#                 # Convert columns to strings temporarily for sorting
#                 distinct_values = distinct_values.astype(str)

#                 # Sort distinct values by the specified columns for hierarchical grouping
#                 distinct_values.sort_values(by=columns, inplace=True)

#                 # Add an empty encoding column for each column in the DataFrame
#                 for col in columns:
#                     distinct_values[f'{col}_enc'] = pd.NA

#                 distinct_values.to_excel(writer, sheet_name=sheet_name, index=False)
                



# def enrich_instacart_df(df):
#     aisles = pd.read_csv('data/input/aisles.csv')
#     products = pd.read_csv('data/input/products.csv')
#     departments = pd.read_csv('data/input/departments.csv')

#     enriched_df = df.merge(products, on='product_id', how='inner') \
#                     .merge(aisles, on='aisle_id', how='inner') \
#                     .merge(departments, on='department_id', how='inner')
#     return enriched_df


from sklearn.metrics import pairwise_distances

def mydistance(x1, x2):
    res = 0
    fix_weight = 1
    for j in range(len(x1)):
        res += fix_weight*abs(x1[j]-x2[j])
    return res

def affinity(X):
    return pairwise_distances(X, metric=mydistance)


import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def evaluate_agglomerative_clustering(X, 
                                      n_clusters_list=[2, 3, 4, 5], 
                                      metric_list=['euclidean', 'manhattan', 'cosine', 'chebyshev', 'hamming'], 
                                      linkage_list=['ward', 'complete', 'average', 'single'],
                                      scale='no',
                                      title='Silhouette Scores for Different Clustering Parameters'):
    # Scale the data if required
    if scale == 'yes':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Store the results
    results = []

    for n_clusters in n_clusters_list:
        for metric in metric_list:
            for linkage in linkage_list:
                if linkage == 'ward' and metric != 'euclidean':
                    continue  # Ward linkage only works with Euclidean distance
                clustering = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
                labels = clustering.fit_predict(X)
                score = silhouette_score(X, labels)
                results.append((n_clusters, metric, linkage, score))

    # Convert results to a structured array for easy plotting
    dtype = [('n_clusters', int), ('metric', 'U10'), ('linkage', 'U10'), ('score', float)]
    results = np.array(results, dtype=dtype)

    # Plot the results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    
    for ax, linkage in zip(axes, linkage_list):
        for metric in metric_list:
            subset = results[(results['metric'] == metric) & (results['linkage'] == linkage)]
            ax.plot(subset['n_clusters'], subset['score'], label=f'{metric}')
            for (n_clusters, score) in zip(subset['n_clusters'], subset['score']):
                ax.text(n_clusters, score, f'{score:.2f}', fontsize=8, ha='right')
        
        ax.set_title(f'Linkage: {linkage}')
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Silhouette Score')
        ax.legend(loc='upper right')
        ax.set_xticks(n_clusters_list)
        ax.set_xticklabels([str(n) for n in n_clusters_list])
    
    fig.suptitle(title, fontsize=16)
    plt.show()


# import seaborn as sns

# def plot_cluster_boxplots(dataframe, features, cluster_labels, file_tag=None):
#     """
#     Plots boxplots for the given features in the dataframe, comparing the specified cluster labels.
#     Optionally adds a prefix to the title of the graph if file_tag is provided.

#     Parameters:
#     dataframe (pd.DataFrame): The dataframe containing the data.
#     features (list): List of features to plot.
#     cluster_labels (list): List of cluster labels to compare.
#     file_tag (str, optional): Prefix to add to the title of the graph.
#     """

#     # Ensure cluster values within each cluster label are ordered
#     for cluster_label in cluster_labels:
#         dataframe[cluster_label] = dataframe[cluster_label].astype('category')
#         dataframe[cluster_label].cat.set_categories(sorted(dataframe[cluster_label].unique()))

#     # Create subplots
#     fig, axes = plt.subplots(nrows=len(features), ncols=len(cluster_labels), figsize=(15, 5 * len(features)))

#     # Loop through each feature and create boxplots
#     for i, feature in enumerate(features):
#         for j, cluster_label in enumerate(cluster_labels):
#             sns.boxplot(x=cluster_label, y=feature, data=dataframe, ax=axes[i, j])
#             title_prefix = f'{file_tag} - ' if file_tag else ''
#             axes[i, j].set_title(f'{title_prefix}{cluster_label} Distribution - {feature}')

#     # Adjust layout
#     plt.tight_layout()
#     plt.show()



from scipy.stats import binom
# from mlxtend.frequent_patterns import fpgrowth, association_rules

# def find_patterns(dataframe, mine_rules=True, min_patterns=10, min_length=3, max_pvalue=0.1, min_confidence=0.6, min_lift=1.4):
#     """
#     Finds frequent itemsets and association rules in the given dataframe using the FP-Growth algorithm.
#     Parameters:
#     dataframe (pd.DataFrame): The input dataframe containing the data to analyze.
#     mine_rules (bool): Whether to mine association rules from the frequent itemsets. Default is True.
#     min_patterns (int): The minimum number of patterns to find before stopping. Default is 10.
#     min_length (int): The minimum length of the itemsets to consider. Default is 3.
#     max_pvalue (float): The maximum p-value for the significance of the patterns. Default is 0.1.
#     min_confidence (float): The minimum confidence for the association rules. Default is 0.6.
#     min_lift (float): The minimum lift for the association rules. Default is 1.4.
#     Returns:
#     pd.DataFrame: A dataframe containing the found patterns and their metrics.
#     """
    
#     def add_significance(patterns, df):
#         N = len(df)
#         probs = {col: df[[col]].eq(1).sum()[col] / N for col in df.columns}
        
#         patterns['significance'] = 0.0
#         for i, pattern in patterns.iterrows():
#             prob = 1
#             for item in pattern['itemsets']:
#                 prob *= probs[item]
#             patterns.at[i, 'significance'] = 1 - binom.cdf(pattern['support'] * N - 1, N, prob)

#     patterns = {}
#     min_support = 1
#     while min_support > 0:
#         min_support = min_support * 0.9
#         print("Finding patterns with min sup %f" % min_support)
#         patterns = fpgrowth(dataframe, min_support=min_support, use_colnames=True)

#         if mine_rules and len(patterns) > 0:
#             patterns = association_rules(patterns, metric="lift", min_threshold=min_lift)
#             patterns = patterns[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
#             patterns = patterns[(patterns['confidence'] >= min_confidence)]
#             patterns['itemsets'] = [x | y for x, y in zip(patterns['antecedents'], patterns['consequents'])]

#         patterns['length'] = patterns['itemsets'].apply(lambda x: len(x))
#         patterns = patterns[(patterns['length'] >= min_length)]
#         add_significance(patterns, dataframe)
#         patterns = patterns[(patterns['significance'] <= max_pvalue)]

#         if len(patterns) >= min_patterns:
#             break

#     patterns['itemsets'] = patterns['itemsets'].apply(lambda x: ', '.join(list(x)))
#     print("Number of found patterns:", len(patterns))
#     return patterns


def timeseries_agg_pivot_df(df, date_col='event_date', nunique_cols=None, sum_cols=None, groupby_cols=None, pivot_cols=['device_category'], include_totals=False):
    """
    Aggregates a time series DataFrame based on specified parameters.

    Args:
    df (pd.DataFrame): The DataFrame to aggregate.
    date_col (str): The name of the event date column.
    nunique_cols (list, optional): List of columns to calculate unique counts.
    sum_cols (list, optional): List of columns to sum.
    groupby_cols (list, optional): List of columns to group by.
    pivot_cols (list): List of columns to pivot.
    include_totals (bool): Whether to include total values per date_col.

    Returns:
    pd.DataFrame: The aggregated DataFrame.
    """
    if nunique_cols is None:
        nunique_cols = []
    if sum_cols is None:
        sum_cols = []
    if groupby_cols is None:
        groupby_cols = []

    if not nunique_cols and not sum_cols:
        raise ValueError("At least one of 'nunique_cols' or 'sum_cols' must be provided.")

    # Group by the specified columns and event date
    grouped = df.groupby(groupby_cols + [date_col] + pivot_cols)

    # Aggregate the unique count and sum columns
    agg_dict = {col: 'nunique' for col in nunique_cols}
    agg_dict.update({col: 'sum' for col in sum_cols})
    aggregated = grouped.agg(agg_dict).reset_index()

    # Pivot the DataFrame based on the pivot columns
    pivoted = aggregated.pivot_table(index=groupby_cols + [date_col], columns=pivot_cols, aggfunc='sum', fill_value=0)

    # Flatten the multi-level columns
    pivoted.columns = [
        f"{'sum_' if col[-1] in sum_cols else 'nunique_'}{'_'.join(map(str, col)).strip()}"
        for col in pivoted.columns.values
    ]
    pivoted.reset_index(inplace=True)

    # Sort by the event date column in ascending order
    pivoted.sort_values(by=date_col, ascending=True, inplace=True)

    if include_totals:
        total_agg_dict = {col: 'nunique' for col in nunique_cols}
        total_agg_dict.update({col: 'sum' for col in sum_cols})
        total_aggregated = df.groupby([date_col]).agg(total_agg_dict).reset_index()
        total_aggregated.columns = [f'total_{col}' if col != date_col else col for col in total_aggregated.columns]
        pivoted = pivoted.merge(total_aggregated, on=date_col, how='left')

    return pivoted



from dslabs_functions import plot_line_chart


def plot_ts_multivariate_chart(data: DataFrame, title: str) -> list[Axes]:
    fig: Figure
    axs: list[Axes]
    height = 10  # Hardcoded height for more margin between subplots
    fig, axs = subplots(data.shape[1], 1, figsize=(3 * height, height / 2 * data.shape[1]))
    fig.suptitle(title)

    for i in range(data.shape[1]):
        col: str = data.columns[i]
        plot_line_chart(
            data[col].index.to_list(),
            data[col].to_list(),
            ax=axs[i],
            xlabel=data.index.name,
            ylabel='',
        )
        axs[i].set_title(col, fontsize=10, pad=10)
        axs[i].margins(y=0.8)  # Add space between each chart
    return axs



from matplotlib.pyplot import subplots, show, gca
from matplotlib.axes import Axes
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from dslabs_functions import HEIGHT, set_chart_labels


def plot_components(
    series: Series,
    title: str = "",
    x_label: str = "time",
    y_label: str = "",
) -> list[Axes]:
    decomposition: DecomposeResult = seasonal_decompose(series, model="add")
    components: dict = {
        "observed": series,
        "trend": decomposition.trend,
        "seasonal": decomposition.seasonal,
        "residual": decomposition.resid,
    }
    rows: int = len(components)
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(rows, 1, figsize=(3 * HEIGHT, rows * HEIGHT))
    fig.suptitle(f"{title}")
    i: int = 0
    for key in components:
        set_chart_labels(axs[i], title=key, xlabel=x_label, ylabel=y_label)
        axs[i].plot(components[key])
        i += 1
    return axs


from dslabs_functions import plot_multiline_chart


def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series




from matplotlib.pyplot import setp
from matplotlib.gridspec import GridSpec


def autocorrelation_study(series: Series, max_lag: int, delta: int = 1):
    k: int = int(max_lag / delta)
    fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values: list = series.tolist()
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        ax.scatter(series.shift(lag).tolist(), series_values)
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
    ax = fig.add_subplot(gs[1, :])
    ax.acorr(series, maxlags=max_lag)
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    return



from matplotlib.axes import Axes
from matplotlib.pyplot import subplots, savefig
from dslabs_functions import PAST_COLOR, FUTURE_COLOR, PRED_PAST_COLOR, PRED_FUTURE_COLOR, HEIGHT


def plot_forecasting_series(
    trn: Series,
    tst: Series,
    prd_tst: Series,
    title: str = "",
    xlabel: str = "time",
    ylabel: str = "",
) -> list[Axes]:
    fig, ax = subplots(1, 1, figsize=(4 * HEIGHT, HEIGHT), squeeze=True)
    fig.suptitle(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.plot(trn.index, trn.values, label="train", color=PAST_COLOR)
    ax.plot(tst.index, tst.values, label="test", color=FUTURE_COLOR)
    ax.plot(prd_tst.index, prd_tst.values, "--", label="test prediction", color=PRED_FUTURE_COLOR)
    ax.legend(prop={"size": 5})

    return ax


from math import sqrt
from dslabs_functions import plot_multibar_chart, FORECAST_MEASURES


def plot_forecasting_eval(trn: Series, tst: Series, prd_trn: Series, prd_tst: Series, title: str = "") -> list[Axes]:
    ev1: dict = {
        "RMSE": [sqrt(FORECAST_MEASURES["MSE"](trn, prd_trn)), sqrt(FORECAST_MEASURES["MSE"](tst, prd_tst))],
        "MAE": [FORECAST_MEASURES["MAE"](trn, prd_trn), FORECAST_MEASURES["MAE"](tst, prd_tst)],
    }
    ev2: dict = {
        "MAPE": [FORECAST_MEASURES["MAPE"](trn, prd_trn), FORECAST_MEASURES["MAPE"](tst, prd_tst)],
        "R2": [FORECAST_MEASURES["R2"](trn, prd_trn), FORECAST_MEASURES["R2"](tst, prd_tst)],
    }

    # print(eval1, eval2)
    fig, axs = subplots(1, 2, figsize=(1.5 * HEIGHT, 0.75 * HEIGHT), squeeze=True)
    fig.suptitle(title)
    plot_multibar_chart(["train", "test"], ev1, ax=axs[0], title="Scale-dependent error", percentage=False)
    plot_multibar_chart(["train", "test"], ev2, ax=axs[1], title="Percentage error", percentage=True)
    return axs


from sklearn.model_selection import train_test_split


# Function to split the series into train and test sets
def series_train_test_split(series, trn_pct=0.90):
    train_size = int(len(series) * trn_pct)
    train, test = series[:train_size], series[train_size:]
    return train, test

from sklearn.base import RegressorMixin

# Define the SimpleAvgRegressor class
class SimpleAvgRegressor:
	def fit(self, series):
		self.mean = series.mean()
	
	def predict(self, series):
		return Series([self.mean] * len(series), index=series.index)



from statsmodels.tsa.arima.model import ARIMA

def fill_missing_values_arima(df, order=(5, 1, 0), freq='D'):
    """
    Fill missing values in a DataFrame using ARIMA.

    Args:
    df (pd.DataFrame): The DataFrame with missing values.
    order (tuple): The (p, d, q) order of the ARIMA model.
    freq (str): The frequency of the time series data.

    Returns:
    pd.DataFrame: The DataFrame with missing values filled.
    """
    df_filled = df.copy()
    
    for column in df.columns:
        series = df[column]
        
        # Check if there are missing values
        if series.isnull().sum() > 0:
            print(f"Filling missing values for column: {column}")
            
            # Ensure the series has the correct frequency
            series = series.asfreq(freq)

            # Fit an ARIMA model
            model = ARIMA(series, order=order)
            model_fit = model.fit()

            # Predict the entire series
            predictions = model_fit.predict(start=0, end=len(series) - 1)

            # Fill missing values with predictions
            series_filled = series.copy()
            series_filled[series.isnull()] = predictions[series.isnull()]

            # Update the filled series in the DataFrame
            df_filled[column] = series_filled

    return df_filled



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dslabs_functions import plot_multibar_chart


def plot_single_model_evaluation(train: Series, test: Series, prd_train: Series, prd_test: Series, title: str = ""):
    # Drop NaN values from the input series
    # train = train.dropna()
    # test = test.dropna()
    # prd_train = prd_train.dropna()
    # prd_test = prd_test.dropna()      
    
    # Calculate performance metrics
    metrics = {
        "MAE": [
            mean_absolute_error(train, prd_train),
            mean_absolute_error(test, prd_test),
        ],
        "MSE": [
            mean_squared_error(train, prd_train),
            mean_squared_error(test, prd_test),
        ],
        "RMSE": [
            np.sqrt(mean_squared_error(train, prd_train)),
            np.sqrt(mean_squared_error(test, prd_test)),
        ],
        "R²": [
            r2_score(train, prd_train),
            r2_score(test, prd_test),
        ]
    }

    # Create a DataFrame from the metrics dictionary
    performance_df = pd.DataFrame(metrics, index=["Train", "Test"]).T
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)
    
    # Plot scale-dependent errors
    scale_dependent_errors = ["MAE", "MSE", "RMSE"]
    scale_dependent_data = performance_df.loc[scale_dependent_errors]
    ax = axs[0]
    ax.set_title("Scale-dependent errors")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    bar_width = 0.2  # Adjusted bar width for more spacing
    index = np.arange(len(scale_dependent_data.columns))
    for i, metric in enumerate(scale_dependent_data.index):
        bars = ax.bar(index + i * bar_width * 1.5, scale_dependent_data.loc[metric], bar_width, label=metric)
        ax.bar_label(bars, fmt='%.2f')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(scale_dependent_data.columns)
    ax.legend()
    
    # Plot percentage errors
    percentage_errors = ["R²"]
    percentage_data = performance_df.loc[percentage_errors]
    ax = axs[1]
    ax.set_title("Percentage errors")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    bar_width = 0.35
    index = np.arange(len(percentage_data.columns))
    for i, metric in enumerate(percentage_data.index):
        bars = ax.bar(index + i * bar_width, percentage_data.loc[metric], bar_width, label=metric)
        ax.bar_label(bars, fmt='%.2f')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(percentage_data.columns)
    ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def plot_performance_metrics(metrics: dict, title: str = ""):
    # Create a DataFrame from the metrics dictionary
    performance_df = pd.DataFrame(metrics)
    
    # Separate scale-dependent errors and percentage errors
    scale_dependent_errors = ["MAE", "MSE", "RMSE"]
    percentage_errors = ["R²"]
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16)
    
    # Plot scale-dependent errors
    scale_dependent_data = performance_df[["Model"] + scale_dependent_errors].set_index("Model").T
    ax = axs[0]
    ax.set_title("Scale-dependent errors")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    bar_width = 0.2  # Adjusted bar width for more spacing
    index = np.arange(len(scale_dependent_data.columns))
    for i, metric in enumerate(scale_dependent_data.index):
        bars = ax.bar(index + i * bar_width * 1.5, scale_dependent_data.loc[metric], bar_width, label=metric)
        ax.bar_label(bars, fmt='%.2f')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(scale_dependent_data.columns)
    ax.legend()
    
    # Plot percentage errors
    percentage_data = performance_df[["Model"] + percentage_errors].set_index("Model").T
    ax = axs[1]
    ax.set_title("Percentage errors")
    ax.set_xlabel("Metrics")
    ax.set_ylabel("Values")
    bar_width = 0.35
    index = np.arange(len(percentage_data.columns))
    for i, metric in enumerate(percentage_data.index):
        bars = ax.bar(index + i * bar_width, percentage_data.loc[metric], bar_width, label=metric)
        ax.bar_label(bars, fmt='%.2f')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(percentage_data.columns)
    ax.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()




print("data_functions loaded")

