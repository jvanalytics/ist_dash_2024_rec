import pandas as pd


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



print("data_functions lodaded")

