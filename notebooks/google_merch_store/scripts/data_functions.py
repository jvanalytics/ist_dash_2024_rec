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
    data: DataFrame, property: str, vars: list[str]
) -> ndarray:
    cols: int = len(vars)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {property}")
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
    trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, metric: str = "accuracy"
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
        title=f"Naive Bayes Models ({metric})",
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
        imp = KNNImputer(n_neighbors=5)
        imp.fit(data)
        ar: ndarray = imp.transform(data)
        df = DataFrame(ar, columns=data.columns, index=data.index)
    return df



from typing import Literal
from numpy import array, ndarray
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import CLASS_EVAL_METRICS, DELTA_IMPROVE, plot_multiline_chart
from dslabs_functions import read_train_test_from_files, plot_evaluation_results

def knn_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, k_max: int=19, lag: int=2, metric='accuracy'
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
    plot_multiline_chart(kvalues, values, title=f'KNN Models ({metric})', xlabel='k', ylabel=metric, percentage=True)

    return best_model, best_params


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
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, d_max: int=10, lag:int=2, metric='accuracy'
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
    print(f'DT best with {best_params['params'][0]} and d={best_params['params'][1]}')
    plot_multiline_chart(depths, values, title=f'DT Models ({metric})', xlabel='d', ylabel=metric, percentage=True)

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
from matplotlib.pyplot import subplots, figure, savefig, show
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
) -> tuple[MLPClassifier | None, dict]:
    nr_iterations: list[int] = [lag] + [
        i for i in range(2 * lag, nr_max_iterations + 1, lag)
    ]

    lr_types: list[Literal["constant", "invscaling", "adaptive"]] = [
        "constant",
        "invscaling",
        "adaptive",
    ]  # only used if optimizer='sgd'
    learning_rates: list[float] = [0.5, 0.05, 0.005, 0.0005]

    best_model: MLPClassifier | None = None
    best_params: dict = {"name": "MLP", "metric": metric, "params": ()}
    best_performance: float = 0.0

    values: dict = {}
    _, axs = subplots(
        1, len(lr_types), figsize=(len(lr_types) * HEIGHT, HEIGHT), squeeze=False
    )
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
                # print(f'MLP lr_type={type} lr={lr} n={nr_iterations[j]}')
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
            title=f"Random Forests with max_depth={d}",
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
            title=f"Gradient Boosting with max_depth={d}",
            xlabel="nr estimators",
            ylabel=metric,
            percentage=True,
        )
    print(
        f'GB best for {best_params["params"][2]} trees (d={best_params["params"][0]} and lr={best_params["params"][1]}'
    )

    return best_model, best_params

print("data_functions lodaded")

