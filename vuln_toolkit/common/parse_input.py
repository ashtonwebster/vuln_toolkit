import pandas as pd
from scipy.io import arff
from sklearn import feature_extraction
import numpy as np
import os

def _create_instance(row):
    """
    Parse individual row of ARFF file in PHP dataset. If the row is from a
    tokens file, the first field must be a string of tokens optionally enclosed
    in single quotes. If the row is from a metrics file, it must be a comma-
    separated list of numbers. The last field must have values 'yes' or 'no,
    which are mapped to the integers 1 and 0, respectively.

    Args:
        row: String representing an individual row of an ARFF file in the PHP
            dataset.

    Returns:
        List of the form [features, label], where features is a list of
        features (tokens or metrics data) and label is an integer having the
        value 1 if the last field of the row is 'yes' and '0' otherwise.
    """

    assert row.count('\'') in [0,2]
    mod_row = row.split('\'');

    # Rows with multiple tokens are wrapped in single quote marks. Otherwise,
    # they are separated by commas.
    if row.count('\'') == 2:
        mod_row = row.split('\'')[1:];
    else:
        mod_row = row.split(',');

    assert (len(mod_row) == 2), "mod row: {}".format(mod_row)    
    return [mod_row[0].replace('\'', ''), 1 if 'yes' in mod_row[1] else 0]

def parse_tokens_file(filename, tfidf_params={"max_features" : 1000}):
    """
    Parse ARFF tokens file and return feature matrix of TF-IDF scores and label
    vector. Note that this function drops null values.

    Args:
        filename: Path to ARFF file with rows of the form (tokens, label),
            where tokens is a string of semicolon-separated tokens optionally
            enclosed in single quote marks; and label is one of 'yes' or 'no'.
        tfidf_params: Parameters passed to TfidfVectorizer. If the max_features
            parameter is not set, the feature matrix may be too large and some
            algorithms may not run in time. By default, the top 1000 tokens by
            descending TF-IDFT score are selected.

    Returns:
        X: DataFrame representing feature matrix. The columns of X are tokens
            and the rows are TF-IDF scores for the tokens in each file.
        y: Series representing label vector. The ith entry of y is 1 if the ith
            row of the file is labeled 'yes' and 0 otherwise.
    """

    f = open(filename)
    file_str = f.readlines()

    # Parse individual rows of ARFF file.
    parsed = [_create_instance(d) for d in file_str[6:]]
    term_list = [x[0] for x in parsed]
    y = pd.Series([x[1] for x in parsed])

    # Apply TF-IDFT transform.
    vectorizer = feature_extraction.text.TfidfVectorizer(**tfidf_params)
    X = vectorizer.fit_transform(term_list)
    X_columns = vectorizer.get_feature_names();
    X_df = pd.DataFrame(X.toarray(), columns=X_columns)
    
    X_df.dropna(inplace=True)
    y = y.ix[X_df.index]
    
    return (X_df, y)

def parse_metrics_file(filename):
    """
    Parse ARFF metrics file and return feature matrix and label vector. Note 
    that this function drops null values.

    Args:
        filename: Path to ARFF file of the form (features, label), where
        features is a comma-separated list of numerical features and label is
        one of 'yes' or 'no'.

    Returns:
        X: DataFarme representing feature matrix. The columns of X are features
            and the rows are values of those features for each file.
        y: Series representing label bector. The ith entry of y is True if the
            ith row of the file is labeled 'yes' and False otherwise.
    """
    
    dataset_dict, dataset_meta = arff.loadarff(filename)

    # Parse features.
    dataset_df = pd.DataFrame(dataset_dict)
    dataset_df = dataset_df.dropna()
    X = dataset_df.iloc[:,0:-1]
    X.reset_index(drop=True, inplace=True)

    # Parse labels.
    f = lambda x : x == "yes"
    y = dataset_df.iloc[:,13].apply(f, 0)

    # Drop null values and reset indices.
    y.reset_index(drop=True,inplace=True)

    return X, y

def _normalize_row(row, mean_arr, std_arr):
    """
    Normalize individual row by computing the standard score (z-score) of each
    feature with respect to mean mean_arr and standard deviation std_arr.

    Args:
        row: Individual row of DataFrame.
        mean_arr: The mean with respect to which standard scores are computed.
        std_arr: The standard deviation with respect to which standard scores
            are computed.

    Returns:
        List of standard scores of entries in row with respect to mean mean_arr
        and standard deviation std_arr.
    """
    
    new_row = []
    for i, f in enumerate(row):
        new_row.append((f - mean_arr[i])/std_arr[i] if std_arr[i] != 0 else 0)

    return new_row

def normalize(X_train, X_test):
    """
    Normalize dataset by computing the standard score (z-score) of the training
    and test sets with respect to the mean and standard deviation of the test
    set. Note that both the training and test sets are normalized using the
    mean and standard deviation of the test set.

    Args:
        X_train: DataFrame representing the training set.
        X_test: DataFrame representing the test set.

    Returns:
        X_train_normalized: Normalized feature matrix for training set.
        X_test_normalized: Normalized feature matrix for test set.
    """
    
    test_mean = list(X_test.apply(np.mean, axis=0))
    test_std = list(X_test.apply(np.std, axis=0))

    X_train_normalized = X_train.apply(
        _normalize_row,
        axis=1,
        mean_arr=test_mean,
        std_arr=test_std
    )
    X_test_normalized = X_test.apply(
        _normalize_row,
        axis=1,
        mean_arr=test_mean,
        std_arr=test_std
    )
    
    return (X_train_normalized, X_test_normalized)

def normalize_per_project(X_train_pool, X_test, test_proj_name, train_project):
    """
    Given a partition of the dataset into training and test sets, standardize
    the training and test sets with respect to the mean and standard deviation
    of each project. See the documentation for split_train_test for detailed
    information about the form of the arguments to this function.

    Args:
        X_train_pool: DataFrame representing the training set.
        X_test: DataFrame representing the test set.
        test_proj_name: The name of the project sampled for testing data.
        train_project: Series the ith entry of which names the project to which
            the ith row of X_train_pool belongs.
    
    Returns:
        X_train_normalized: Normalized feature matrix for training set.
        X_test_normalized: Normalized feature matrix for test set.
    """
    
    X_train_norm = pd.DataFrame()
    for project in set(train_project):
        bitmask = train_project == project
        X_for_proj = X_train_pool[bitmask]
        X_norm, _ = normalize(X_for_proj,X_for_proj)
        X_train_norm = X_train_norm.append(X_norm)

    X_test_norm, _ = normalize(X_test, X_test) 
    return X_train_norm, X_test_norm

def _get_metric_filenames():
    """
    Return full paths of metrics files in PHP dataset (Moodle, phpMyAdmin and
    Drupal). The VULN_INPUT environment variable must be set for this function
    to work correctly (see README for more information).

    Returns:
        List of full paths to metrics files in PHP dataset.
    """
    
    input_dir = os.environ["VULN_INPUT"]
    return [input_dir + filename for filename in [
        '/drupal-6_0/drupal-6_0-metrics.arff', \
        '/moodle-2_0_0/moodle-2_0_0-metrics.arff', \
        '/phpmyadmin-3_3_0/phpmyadmin-3_3_0-metrics.arff'
    ]]

def _get_token_filenames():
    """
    Return full paths of tokens files in PHP dataset (Moodle, phpMyAdmin and
    Drupal). The VULN_INPUT environment variable must be set for this function
    to work correctly (see README for more information).

    Returns:
        List of full paths to tokens files in PHP dataset.
    """
    
    input_dir = os.environ["VULN_INPUT"]
    return [input_dir + filename for filename in [
        '/drupal-6_0/drupal-6_0-tokens.arff', \
        '/moodle-2_0_0/moodle-2_0_0-tokens.arff', \
        '/phpmyadmin-3_3_0/phpmyadmin-3_3_0-tokens.arff'
    ]]

def parse_promise_csvs():
    """
    Parse files in PROMISE dataset. Note that the number of defects per file is
    mapped to an integer whose value is 1 if some defect is present and 0
    otherwise. The  VULN_INPUT environment variable must be set for this
    function to work correctly (see README for more information)

    Returns:
        List of dictionaries with keys 'proj', the project name; 'X', a
        DataFrame of metrics data; and 'y', a Series of labels.
    """
    
    all_csv_dir = os.environ["VULN_INPUT"] + '/ck/all_csv/'
    data_dict_list = []
    for filename in os.listdir(all_csv_dir):
        df = pd.read_csv(all_csv_dir + filename)
        proj_name = filename.replace(".csv","")
        X = df.loc[:, "wmc":"avg_cc"]
        y = df.bug.apply(lambda x: 0 if x==0 else 1)
        assert len(X) == len(y)
        data_dict = {"X" : X, "y" : y, 'proj' : proj_name}
        data_dict_list.append(data_dict)
    return data_dict_list

def parse_metric_projects():
    """
    Parse metrics files in PHP dataset (Moodle, phpMyAdmin and Drupal). The
    VULN_INPUT environment variable must be set for this function to work
    correctly (see README for more information).

    Returns:
        List of dictionaries with keys 'proj', the project name; 'X', a
        DataFrame of metrics data; and 'y', a Series of labels.
    """
    
    data_dict = []
    for filename in _get_metric_filenames():
        X, y = parse_metrics_file(filename)
        name_clean = filename.split('/')[-1].split('-')[0]
        data_dict.append({'X' : X, 'y': y, 'proj' : name_clean})
    return data_dict

def parse_token_projects():
    """
    Parse tokens files in PHP dataset (Moodle, phpMyAdmin and Drupal). The
    VULN_INPUT environment variable must be set for this function to work
    correctly (see README for more information).

    Returns:
        List of dictionaries with keys 'proj', the project name; 'X', a
        DataFrame of tokens data (by default, the TF-IDFT score for the top
        1000 tokens by descending TF-IDF score); and 'y', a Series of labels.
    """
    
    data_dict = []
    for filename in _get_token_filenames():
        X, y = parse_tokens_file(filename)
        name_clean = filename.split('/')[-1].split('-')[0]
        data_dict.append({'X' : X, 'y': y, 'proj' : name_clean})
    return data_dict

def split_train_test(data_dict, target_proj_name, target_proj_test_sample,
                     rand_seed=None, fillna=None):
    """
    Partition dataset into training and test sets. A specified proportion of
    the project named by target_proj_name is reserved as testing data; all
    remaining data, including the remaining portion of the project sampled for
    testing data, is used as training data.

    Args:
        data_dict: A list of dictionaries with keys 'proj', then the name of
            the project; 'X', A DataFrame of features (metrics or tokens); and
            'y', a Series of labels.
        target_proj_name: The name of the project to sample for testing data.
        target_proj_test_sample: The proportion of data from the project named
            by target_proj_name to sample for testing data. For example, a
            value of 0.8 means that 80% of the named project will be sampled
            for testing data; the remaining 20% will be used as training data.
        rand_seed: Random seed passed to DataFrame.sample (default = None).
        fillna: Value passed to DataFrame.fillna (default = None).

    Returns:
        X_train_pool: DataFrame representing feature matrix for training set.
        y_train_pool: Series representing label vector for training set.
        proj_train_pool: Series of project names, the ith entry of which names
            the project to which the ith row in X_train_pool and y_train_pool
            belongs.
        X_test: DataFrame representing feature matrix for test set.
        y_test: Series representing label vector for test set.
    """

    # Combine projects into single DataFrame.
    master_df = pd.DataFrame()
    for d in data_dict:
        df = d['X']
        df['y'] = d['y']
        df['proj'] = [d['proj']] * df.shape[0]
        master_df = master_df.append(df)

    # Reset row indices to start at 1.
    master_df.reset_index(drop=True, inplace=True)
    
    # Set missing tokens to 0.
    if not (fillna is None):
        master_df.fillna(value=fillna, inplace=True)

    # Sample target project for testing data.
    test_df = master_df[master_df['proj'] == target_proj_name].sample(
        frac=target_proj_test_sample,
        random_state=rand_seed
    )

    # Remove sampled data from training data.
    master_df = master_df.ix[master_df.index.difference(test_df.index),]

    # Reset row indices to start at 1.
    master_df.reset_index(drop=True, inplace=True)

    y_train_pool = master_df['y']
    proj_train_pool = master_df['proj']
    X_train_pool = master_df.drop('y', axis=1).drop('proj', axis=1)
    y_test = test_df['y']
    X_test = test_df.drop('y', axis=1).drop('proj', axis=1)

    return (X_train_pool, y_train_pool, proj_train_pool, X_test, y_test)
