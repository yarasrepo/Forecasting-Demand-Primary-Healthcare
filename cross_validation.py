import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
import itertools as it
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from scipy.stats.stats import pearsonr, spearmanr
import pandas as pd
import scipy.special
import scipy.stats


class LearningModel:
    def __init__(self, df, target_variable, split_ratio: float,
                    output_folder,
                    cols_drop=None, scale=True, scale_input=True, scale_output=False,
                    output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                    input_zscore=None, input_minmax=None, input_box=None, input_log=None,
                    service_name=None, mohafaza=None, testing_data=None,
                    grid=True, random_grid=False,
                    nb_folds_grid=None, nb_repeats_grid=None, nb_folds_random=None,
                    nb_repeats_random=None, nb_iterations_random=None,
                    loo=False,
                    save_errors_xlsx=True,
                    save_validation=False):

        # data frames. If df_test is None, df will be split into training and testing according to split_ratio
        # Otherwise, df will be training, df_test will be testing
        self.df = df
        self.testing_data = testing_data

        # drop NaN values
        self.df = self.df.dropna()
        if self.testing_data is not None:
            self.testing_data = self.testing_data.dropna()

        if self.testing_data is None:
            nb_rows_test = int(round(len(self.df) * split_ratio))
            nb_rows_train = len(self.df) - nb_rows_test

            self.df_train = self.df[0: nb_rows_train]
            self.df_test = self.df[nb_rows_train:]
            print('first %d rows for training, last %d rows for testing' % (nb_rows_train, nb_rows_test))
        else:
            self.df_train = self.df
            self.df_test = self.testing_data
            print('param df is the training, param df_test is the testing ...')

        # original testing data (the testing data before dropping columns from it)
        # needed when attaching the 'predicted' column
        self.df_test_orig = self.df_test

        # output folder to save plots and data
        self.output_folder = output_folder

        # save the training and testing datasets before doing anything
        self.save_train_test_before_modeling()

        self.target_variable = target_variable

        # list of columns to drop
        self.cols_drop = cols_drop
        if self.cols_drop is not None:
            self.df_train = self.df_train.drop(self.cols_drop, axis=1)
            self.df_test = self.df_test.drop(self.cols_drop, axis=1)
            print('list of columns used in modeling')
            print(list(self.df_test.columns.values))

        print('shuffling the 80% training before cv ...')
        self.df_train = self.df_train.sample(frac=1, random_state=42)

        # output folder
        self.output_folder = output_folder

        # scaling input & output
        self.scale = scale
        self.scale_input = scale_input
        self.scale_output = scale_output

        # specify scaling method for output
        self.output_zscore = output_zscore
        self.output_minmax = output_minmax
        self.output_box = output_box
        self.output_log = output_log

        # specify scaling method for input
        self.input_zscore = input_zscore
        self.input_minmax = input_minmax
        self.input_box = input_box
        self.input_log = input_log

        # related to cross validation
        self.grid = grid
        self.random_grid = random_grid
        self.nb_folds_grid = nb_folds_grid
        self.nb_repeats_grid = nb_repeats_grid
        self.nb_folds_random = nb_folds_random
        self.nb_repeats_random = nb_repeats_random
        self.nb_iterations_random = nb_iterations_random
        self.loo = loo
        self.split_ratio = split_ratio

        # save error metrics to xlsx sheet
        self.save_errors_xlsx = save_errors_xlsx
        self.save_validation = save_validation

        if self.save_errors_xlsx:
            self.results = pd.DataFrame(columns=['r2', 'adj-r2', 'rmse', 'mse', 'mae', 'mape',
                                                     'avg_%s' % self.target_variable,
                                                     'pearson', 'spearman', 'distance', 'coefficients'])
        else:
            self.results = None

        if self.save_validation:
            self.results_validation = pd.DataFrame(columns=['r2', 'adj-r2',
                                                            'rmse', 'mse', 'mae', 'mape'])
        else:
            self.results_validation = None

        # if grid_search=True and random_search_then_grid=True
        if self.grid and self.random_grid:
            raise ValueError('you cannot set both `grid` and `random_grid` to True. Either one must be False')

        # if grid=False and random_search_then_grid_search=False
        elif not self.grid and not self.random_grid:
            raise ValueError('you cannot set both `grid` and `random_grid` to False. Either one must be True')

        elif self.grid and not self.random_grid:
            if self.nb_folds_grid is None:
                raise ValueError('Please set nb_folds_grid to a number')
        else:
            if self.nb_iterations_random is None or self.nb_folds_random is None:
                raise ValueError('Please specify\n1.nb_iterations_random\n'
                                 '2.nb_folds_random\n3.nb_repeats_random(if needed)')

        # service_name & mohafaza for MoPH
        self.service_name = service_name
        self.mohafaza = mohafaza

        df_without_target = self.df_train
        df_without_target = df_without_target.drop([self.target_variable], axis=1)
        self.feature_names = list(df_without_target.columns.values)
        print(self.feature_names)

        # numpy arrays X_train, y_train, X_test, y_test
        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.target_variable])

        # get the list of indices of columns for each scaling type
        self.idx_zscore, self.idx_minmax, self.idx_box, self.idx_log = None, None, None, None

        if self.input_zscore is not None:
            self.idx_zscore = list(range(self.input_zscore[0], self.input_zscore[1]))

        if self.input_minmax is not None:
            self.idx_minmax = list(range(self.input_minmax[0], self.input_minmax[1]))

        if self.input_box is not None:
            self.idx_box = list(range(self.input_box[0], self.input_box[1]))

        if self.input_log is not None:
            self.idx_log = list(range(self.input_log[0], self.input_log[1]))

        cols_coefs = self.feature_names + ['y_intercept']
        self.coefficients = pd.DataFrame(columns=cols_coefs)

    def save_train_test_before_modeling(self):
        ''' save the training and testing data frames before any processing happens to them '''
        path = self.output_folder + 'train_test_before_modeling/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.df_train.to_csv(path + 'training.csv', index=False)
        self.df_test.to_csv(path + 'testing.csv', index=False)

    def cross_validation(self, model_used, hyperparams, model_name):

        # if grid=True and random_grid=False
        if self.grid and not self.random_grid:
            self.cross_validation_grid(model_used, hyperparams, model_name)

        # if grid=False and random_grid=True
        else:
            self.cross_validation_random_grid(model_used, hyperparams, model_name)

    def inverse_boxcox(self, y_box, lambda_):
        pred_y = np.power((y_box * lambda_) + 1, 1 / lambda_) - 1
        return pred_y



    def create_output_dataset(self, y_pred, model_name, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # JUST TO AVOID THE CURRENT BUG
        df_test_curr = self.df_test_orig
        if 'predicted' in list(df_test_curr.columns.values):
            df_test_curr = df_test_curr.drop('predicted', axis=1)

        # add the predicted value to the df
        target_loc = df_test_curr.columns.get_loc(self.target_variable)
        df_test_curr.insert(target_loc + 1, 'predicted', list(y_pred))

        if self.service_name is None and self.mohafaza is None:
            # df_train.to_csv('train_df.csv')
            df_test_curr.to_csv(output_folder + 'test_df_%s.csv' % model_name, index=False)
        else:
            # df_train.to_csv('%s_%s_train.csv' % (service_name, mohafaza))
            if not os.path.exists(output_folder + '/%s_%s/' % (self.service_name, self.mohafaza)):
                os.makedirs(output_folder + '/%s_%s/' % (self.service_name, self.mohafaza))
            df_test_curr.to_csv(output_folder + '%s_%s/test_%s.csv' % (self.service_name, self.mohafaza, model_name))

        return df_test_curr


    def produce_learning_curve(self, model, model_name, nb_splits, output_folder, parameters, nb_repeats=None):

        '''
        produce learning curve of a certain model, using either KFold or repeated KFold cross validation
        :param model: the model
        :param model_name: name of the model, string.
        :param nb_splits: number of splits in KFold
        :param output_folder: path to output folder. If doesn't exist, will be created at runtime
        :param nb_repeats: number of repeats in case of RepeatedKFold. By defualt None. If None,
        KFold will be used instead
        :return: saves the learning curve
        '''

        X_train, y_train = self.X_train, self.y_train
        pipe = None

        if self.scale:
            if self.scale_output:
                if self.output_zscore:
                    scaler = StandardScaler()
                    y_train = scaler.fit_transform(y_train)
                elif self.output_minmax:
                    scaler = MinMaxScaler()
                    y_train = scaler.fit_transform(y_train)
                elif self.output_log:
                    y_train = np.log(y_train)
                else:
                    y_train, _ = scipy.stats.boxcox(y_train)

            if self.scale_input:
                if self.input_zscore is not None and self.input_minmax is not None:
                    # print('1st condition')
                    ct = ColumnTransformer([('standard', StandardScaler(), self.idx_zscore),
                                            ('minmax', MinMaxScaler(), self.idx_minmax)], remainder='passthrough')
                    pipe = Pipeline(steps=[('preprocessor', ct), ('model', model(**parameters))])

                elif self.input_zscore is not None and self.input_minmax is None:
                    # print('2nd condition')
                    ct = ColumnTransformer([('standard', StandardScaler(), self.idx_zscore)], remainder='passthrough')
                    pipe = Pipeline(steps=[('preprocessor', ct), ('model', model(**parameters))])

                elif self.input_zscore is None and self.input_minmax is not None:
                    # print('3rd condition')
                    ct = ColumnTransformer([('minmax', MinMaxScaler(), self.idx_minmax)], remainder='passthrough')
                    pipe = Pipeline(steps=[('preprocessor', ct), ('model', model(**parameters))])

                else:
                    # print('4th condition')
                    pipe = model(**parameters)

        else:
            # print('4th condition')
            pipe = model(**parameters)

        if nb_repeats is None:
            cv = KFold(n_splits=nb_splits, random_state=2652124)
        else:
            cv = RepeatedKFold(n_splits=nb_splits, n_repeats=nb_repeats, random_state=2652124)

        # if box or log transform is needed, this must not necessarily be done in a pipeline manner
        # because same transformation is done for training and validation, UNLIKE z-score and minmax
        # whereby scaling must be done on training THEN taking the parameters and apply them
        # to the validation

        if self.scale:
            if self.scale_input:
                if self.input_box is not None:
                    # apply BoxCox transform to the specified columns.
                    if X_train.dtype != 'float64':
                        X_train = X_train.astype('float64')

                    X_train_boxscaled = np.array([list(scipy.stats.boxcox(X_train[:, self.idx_box[i]])[0]) for i in range(len(self.idx_box))]).T

                    for i in range(len(self.idx_box)):
                        X_train[:, self.idx_box[i]] = X_train_boxscaled[:, i]

                if self.input_log is not None:
                    # apply Log transform to the specified columns.
                    if X_train.dtype != 'float64':
                        X_train = X_train.astype('float64')

                    X_train_logscaled = np.log(X_train[:, self.idx_log])

                    for i in range(len(self.idx_log)):
                        X_train[:, self.idx_log[i]] = X_train_logscaled[:, i]

        train_sizes, train_scores, test_scores = learning_curve(pipe, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')  # calculate learning curve values

        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)

        plt.figure()
        plt.xlabel("Number of Training Samples")
        plt.ylabel("MSE")

        plt.plot(train_sizes, train_scores_mean, label="training")
        plt.plot(train_sizes, test_scores_mean, label="validation")
        plt.legend()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + '%s_learning_curve.png' % model_name)
        plt.close()

    def plot_actual_vs_predicted(self, df, model_name, output_folder, predicted_variable):

        plt.plot(list(range(1, len(df) + 1)), df[self.target_variable], color='b', label='actual')
        plt.plot(list(range(1, len(df) + 1)), df[predicted_variable], color='r', label='predicted')
        plt.legend(loc='best')
        # plt.suptitle('actual vs. predicted forecasts for %s in %s' % (self.service_name, self.mohafaza))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + 'forecasts_%s' % model_name)
        plt.close()

    def plot_actual_vs_predicted_scatter_bisector(self, df, model_name, output_folder, predicted_variable,):
        fig, ax = plt.subplots()

        ax.scatter(df[self.target_variable], df[predicted_variable], c='black')

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # plt.suptitle('actual vs. predicted forecasts')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + 'scatter_%s' % model_name)
        plt.close()

    def errors_to_csv(self):
        ''' saves the error metrics (stored in `results`) as a csv file '''
        if self.results is not None:
            errors_df = self.results
            errors_df = errors_df.sort_values(by=['rmse'])
            path = self.output_folder + 'error_metrics_csv/'
            if not os.path.exists(path):
                os.makedirs(path)
            errors_df.to_csv(path + 'errors.csv')
            if self.results_validation is not None:
                validation_errors_df = self.results_validation
                validation_errors_df = validation_errors_df.sort_values(by=['rmse'])
                validation_errors_df.to_csv(path + 'errors_validation.csv')


def create_output_dataset(df_test_curr, y_pred, model_name, output_folder, target_variable, service_name=None, mohafaza=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # add the predicted value to the df
    target_loc = df_test_curr.columns.get_loc(target_variable)
    df_test_curr.insert(target_loc + 1, 'predicted', list(y_pred))

    if service_name is None and mohafaza is None:
        # df_train.to_csv('train_df.csv')
        df_test_curr.to_csv(output_folder + 'test_df_%s.csv' % model_name, index=False)
    else:
        # df_train.to_csv('%s_%s_train.csv' % (service_name, mohafaza))
        if not os.path.exists(output_folder + '/%s_%s/' % (service_name, mohafaza)):
            os.makedirs(output_folder + '/%s_%s/' % (service_name, mohafaza))
        df_test_curr.to_csv(output_folder + '%s_%s/test_%s.csv' % (service_name, mohafaza, model_name))

    return df_test_curr


# def mean_absolute_percentage_error(y_true, y_pred):
#     '''
#     Function to compute the mean absolute percentage error (MAPE) between an actual and
#     predicted vectors
#     :param y_true: the actual values
#     :param y_pred: the predicted values
#     :return: MAPE
#     '''
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_stats(y_test, y_pred, nb_columns):
    '''
    Function to compute regression error metrics between actual and predicted values +
    correlation between both using different methods: Pearson, Spearman, and Distance
    :param y_test: the actual values. Example df['actual'] (the string inside is the name
    of the actual column. Example: df['LE (mm)'], df['demand'], etc.)
    :param y_pred: the predicted vlaues. Example df['predicted']
    :param nb_columns: number of columns <<discarding the target variable column>>
    :return: R2, Adj-R2, RMSE, MSE, MAE, MAPE
    '''

    if not isinstance(y_test, list):
        y_test = list(y_test)
    if not isinstance(y_pred, list):
        y_pred = list(y_pred)

    n = len(y_test)
    r2_Score = r2_score(y_test, y_pred) # r-squared
    adjusted_r2 = 1 - ((1 - r2_Score) * (n - 1)) / (n - nb_columns - 1) # adjusted r-squared
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE
    mse_score = mean_squared_error(y_test, y_pred) # MSE
    mae_score = mean_absolute_error(y_test, y_pred) # MAE
    mape_score = mean_absolute_percentage_error(y_test, y_pred) # MAPE

    if len(y_test) >= 2:
        pearson_corr, _ = pearsonr(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        distance_corr = distance.correlation(y_test, y_pred)

        return r2_Score, adjusted_r2, rmse_score, mse_score, mae_score, mape_score, pearson_corr, spearman_corr, distance_corr
    else:

        return r2_Score, adjusted_r2, rmse_score, mse_score, mae_score, mape_score


def check_service_mohafaza(file_name):
    services = ['General Medicine', 'Gynaecology', 'Pediatrics', 'Pharmacy']
    mohafazas = ['akkar', 'bikaa', 'Tripoli']
    curr_service, curr_mohafaza, curr_datasubset = None, None, None
    for service in services:
        for mohafaza in mohafazas:
            # if the service and mohafaza are substring of the file's path
            if service in file_name and mohafaza in file_name:
                curr_service = service
                curr_mohafaza = mohafaza
                curr_datasubset = '%s_%s' % (curr_service, curr_mohafaza)
                print('reading %s in %s data subset ... ' % (curr_service, curr_mohafaza))

    return curr_service, curr_mohafaza, curr_datasubset