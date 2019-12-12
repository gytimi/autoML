import config
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from math import sqrt
import warnings

# warnings.filterwarnings("ignore")


def my_reader(filename, sheetname='Sheet1', separ=','):
    global df_read
    filename_list = filename.split('.')
    extension = filename_list[-1]
    if extension == 'csv':
        df_read = pd.read_csv(filename, sep=separ)
    if extension == 'data':
        df_read = pd.read_csv(filename, sep=separ, header=None)
    if extension == 'txt':
        df_read = pd.read_csv(filename, sep=separ)
    if extension == 'json':
        df_read = pd.read_json(filename)
    if extension == 'html':
        df_read = pd.read_html(filename)
    if extension == 'xls':
        df_read = pd.read_excel(pd.ExcelFile(filename), sheetname)
    if extension == 'xlsx':
        df_read = pd.read_excel(pd.ExcelFile(filename), sheetname)
    if extension == 'feather':
        df_read = pd.read_feather(filename)
    if extension == 'parquet':
        df_read = pd.read_parquet(filename)
    if extension == 'msg':
        df_read = pd.read_msgpack(filename)
    if extension == 'dta':
        df_read = pd.read_stata(filename)
    if extension == 'sas7bdat':
        df_read = pd.read_sas(filename)
    if extension == 'pkl':
        df_read = pd.read_pickle(filename)
    return df_read


def my_train_test_split(act_my_data, act_test_size=0.5):
    act_train_df, act_test_df = train_test_split(act_my_data,
                                                 test_size=act_test_size)
    return act_train_df, act_test_df


def to_dummies(to_dummy_data):
    if config.to_dummies:
        for col in to_dummy_data.columns:
            unique_col_num = len(pd.unique(to_dummy_data[col]))
            dummy_max = int(len(to_dummy_data[col]) / 10)
            col_type = to_dummy_data.dtypes[col]
            if (col_type == "object") \
                    & (unique_col_num < dummy_max) \
                    & (unique_col_num > 1):
                temp_dummies = pd.get_dummies(to_dummy_data[col])
                to_dummy_data = pd.concat([to_dummy_data, temp_dummies],
                                          axis=1, sort=False)
    return to_dummy_data


def to_pure_numbers(mydata):
    num_type = (mydata.dtypes == "float64") | (mydata.dtypes == "int64")
    number_list = list((mydata.dtypes[num_type]).keys())
    number_list.remove(config.target)
    return number_list


def guess_goal(mydata, target):
    cardin = dict(mydata.apply(pd.Series.nunique))
    target_type = mydata.dtypes[target]
    if(target_type == 'float64') | (target_type == 'int64'):
        if cardin[target] > 50:
            act_regression = True
            act_classification = False
        else:
            act_regression = False
            act_classification = True
    else:
        act_regression = False
        act_classification = True
    return act_regression, act_classification


class MissingValueHandle(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, transform_data):
        if config.missing_bool:
            desc_df = transform_data.describe()
            desc_col = desc_df.columns
            for c in desc_col:
                if config.missing_value_handle == 'min-1':
                    transform_data[c].fillna(desc_df[c][3] - 1, inplace=True)
                if config.missing_value_handle == 'mean':
                    transform_data[c].fillna(desc_df[c][1], inplace=True)
        return transform_data


class DuplicatedRowHandle(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, transform_data):
        if config.duplicated_bool:
            valami = ~transform_data.duplicated()
            transform_data = transform_data[valami]
        return transform_data


class MyMinMaxScaler(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, transform_data):
        if config.min_scaler_bool:
            desc_df = transform_data.describe()
            desc_col = desc_df.columns
            scaler = preprocessing.MinMaxScaler()
            scaled_df = scaler.fit_transform(transform_data)
            transform_data = pd.DataFrame(scaled_df, columns=desc_col)
        return transform_data


class Standardize(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, transform_data):
        if config.standardize_bool:
            desc_df = transform_data.describe()
            desc_col = desc_df.columns
            scaler = preprocessing.StandardScaler()
            scaled_df = scaler.fit_transform(transform_data)
            transform_data = pd.DataFrame(scaled_df, columns=desc_col)
        return transform_data


class Digitize(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, transform_data):
        if len(config.discretize) > 0:
            for col in config.discretize:
                to_digit = transform_data[col].to_numpy()
                to_digit_min = int(transform_data[col].min())
                to_digit_max = int(transform_data[col].max())
                to_digit_step = int((to_digit_max - to_digit_min) / 10)
                bins = np.arange(to_digit_min, to_digit_max + 1, to_digit_step)
                transform_data[col] = pd.Series(np.digitize(to_digit, bins))
        return transform_data


def my_evaluation_pipe(y_pred, y_true):
    act_list = []
    act_eval = []
    act_column = []
    if classification:
        y_true = y_true.array
        act_column = ['Accuracy', 'Precision', 'Recall', 'F1']
        # Accuracy
        act_eval.append(accuracy_score(y_true, y_pred))
        try:
            # Precision
            act_eval.append(precision_score(y_true, y_pred))
            # Recall
            act_eval.append(recall_score(y_true, y_pred))
            # F1
            act_eval.append(f1_score(y_true, y_pred))
        except:
            # Precision
            act_eval.append(precision_score(y_true, y_pred, average=None))
            # Recall
            act_eval.append(recall_score(y_true, y_pred, average=None))
            # F1
            act_eval.append(f1_score(y_true, y_pred, average=None))
    if regression:
        act_column = ['Mean Absolute Error', 'Mean Squarred Error',
                      'R2 Score', 'Explained Variance score']
        # Mean Absolute Error
        act_eval.append(mean_absolute_error(y_true, y_pred))
        # Mean Squarred Error
        act_eval.append(mean_squared_error(y_true, y_pred))
        # R2 Score
        act_eval.append(r2_score(y_true, y_pred))
        # Explained Variance score
        act_eval.append(explained_variance_score(y_true, y_pred))
    act_list.append(act_eval)
    return pd.DataFrame(act_list, columns=act_column)


def is_binary(incoming_data):
    binary_cols = []
    for col in incoming_data.columns:
        unique_col_num = len(pd.unique(incoming_data[col]))
        if unique_col_num == 2:
            binary_cols.append(col)
    return binary_cols


def is_unary(incoming_data):
    unary_cols = []
    for col in incoming_data.columns:
        unique_col_num = len(pd.unique(incoming_data[col]))
        if unique_col_num == 1:
            unary_cols.append(col)
    return unary_cols


def is_time(incoming_data):
    time_cols = []
    for col in incoming_data.columns:
        try:
            s = incoming_data[col].astype(str)
            for s1 in s:
                temp_series = pd.Timestamp(s1)
            time_cols.append(col)
        except:
            pass
    return time_cols


def is_outlier(incoming_data):
    outlier_df = incoming_data[incoming_data.apply(lambda x: np.abs(x - x.mean()) / x.std() > 3).all(axis=1)]
    return not outlier_df.empty


def auto_ml():
    # Reading from file
    my_data = my_reader(config.filename, separ=config.file_separ)

    # Binary and Unary columns search
    is_binary_list = is_binary(my_data)
    is_unary_list = is_unary(my_data)

    # Time columns search
    is_time_list = is_time(my_data)

    # To dummy
    my_data = to_dummies(my_data)

    # Train-test split
    train_df, test_df = my_train_test_split(my_data, act_test_size=config.test_size)

    # Pure numbers will be the input variables
    input_vars = to_pure_numbers(my_data)

    # Choosing if it is a regression or classification
    global regression, classification
    regression, classification = guess_goal(my_data, config.target)

    # Modelling and building the pipeline
    n_neighbors = 15
    X = train_df[input_vars]
    if regression:
        pipe_1 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', LinearRegression(fit_intercept=True))])
        pipe_2 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model',
                           neighbors.KNeighborsRegressor(n_neighbors,
                                                         weights='distance'))])
        pipe_3 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', linear_model.BayesianRidge())])
        pipe_4 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', linear_model.SGDRegressor())])
        pipe_5 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', linear_model.ElasticNet())])
        pipe_6 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', linear_model.Ridge())])
        pipe_7 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', linear_model.Lasso())])
        pipe_8 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', RandomForestRegressor(max_depth=2,
                                                          random_state=0,
                                                          n_estimators=100))])
        pipe_dict = {0: 'LinearRegression',
                     1: 'KNeighborsRegressor',
                     2: 'BayesianRidge',
                     3: 'SGDRegressor',
                     4: 'ElasticNet',
                     5: 'Ridge',
                     6: 'Lasso',
                     7: 'RandomForestRegressor'}

    if classification:
        pipe_1 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                           ('model', LogisticRegression(random_state=42))])
        pipe_2 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                           ('model', neighbors.KNeighborsClassifier(n_neighbors))])
        pipe_3 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', RandomForestClassifier(n_estimators=100,
                                                           max_depth=2,
                                                           random_state=0))])
        pipe_4 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', linear_model.SGDClassifier())])
        pipe_5 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', MLPClassifier())])
        pipe_6 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', GradientBoostingClassifier())])
        pipe_7 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', GaussianNB())])
        pipe_8 = Pipeline([('missing', MissingValueHandle()),
                           ('duplicated', DuplicatedRowHandle()),
                           ('discretize', Digitize()),
                           ('standardize', Standardize()),
                           ('minmaxscaler', MyMinMaxScaler()),
                          ('model', SVC(gamma='auto'))])
        pipe_dict = {0: 'LogisticRegression',
                     1: 'KNeighborsClassifier',
                     2: 'RandomForestClassifier',
                     3: 'SGDClassifier',
                     4: 'MLPClassifier',
                     5: 'GradientBoostingClassifier',
                     6: 'GaussianNB',
                     7: 'SVC'}

    # List of pipelines
    pipelines = [pipe_1, pipe_2, pipe_3, pipe_4, pipe_5, pipe_6, pipe_7, pipe_8]

    # Fit the pipelines
    for pipe in pipelines:
        pipe.fit(train_df[input_vars], train_df[config.target])

    # Is there outlier
    outlier_bool = is_outlier(X)

    corr_df = X.corr()

    # Open new file
    result_path = './test_eval/Result_params_' +\
                  str(config.filename.split("/")[-1].split(".")[0]) + '.txt'
    result_file = open(result_path, 'w')
    result_file.write("Filename: " + str(config.filename) + '\n')
    result_file.write("Target: " + str(config.target) + '\n')
    if regression:
        result_file.write("Prediction type: Regression" + '\n')
    else:
        result_file.write("Prediction type: Classification" + '\n')
    result_file.write("Test size: " + str(config.test_size*100) + "%" + '\n')
    result_file.write("Model input columns: " + str(input_vars) + '\n')
    result_file.write("Used preparations: " + '\n')
    if config.missing_bool:
        result_file.write("Missing value handle (" +
                          str(config. missing_value_handle) +
                          "), ")
    if config.min_scaler_bool:
        result_file.write("Min scaling, ")
    if config.standardize_bool:
        result_file.write("Standardize, ")
    if config.to_dummies:
        result_file.write("To dummies")
    result_file.write('\n' + "Discretize columns: " + str(config.discretize) + '\n')
    result_file.write("Binary columns: " + str(is_binary_list) + '\n')
    result_file.write("Unary columns: " + str(is_unary_list) + '\n')
    result_file.write("Time columns: " + str(is_time_list) + '\n')
    if outlier_bool:
        result_file.write("There is outlier in the data." + '\n')

    # Evaluation
    result_df = pd.DataFrame()
    result_cols = []
    for idx, val in enumerate(pipelines):
        result_df = pd.concat([result_df,
                               my_evaluation_pipe(val.predict(test_df[input_vars]),
                                                  test_df[config.target])])
        result_cols.append(pipe_dict[idx])

    result_df.index = result_cols
    result_file.close()

    with pd.ExcelWriter("./test_eval/Evaluation_"
                        + str(config.filename.split("/")[-1].split(".")[0])
                        + ".xlsx") as writer:
        if regression:
            result_df.to_excel(writer, sheet_name="Regression")
        else:
            result_df.to_excel(writer, sheet_name="Classification")
        corr_df.to_excel(writer, sheet_name="Correlation")


auto_ml()
