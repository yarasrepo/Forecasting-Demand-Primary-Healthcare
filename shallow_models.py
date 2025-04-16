import pandas as pd
import numpy as np
import os
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    fbeta_score, roc_curve, auc, precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score
from imblearn.metrics import geometric_mean_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
import pickle
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, learning_curve, RandomizedSearchCV, StratifiedKFold, cross_validate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing
from models_hyperparams_grid import *


def get_results(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    gmean = geometric_mean_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    print('F2: {:.3f}\n G-mean: {:.3f}\nROC-AUC: {:.3f}\nAccuracy: {:.3f}\nPrecision: {:.3f}\nRecall: {:.3f}\nF1: {:.3f}\n==============================================\n'.format(
        f2, gmean, roc_auc, acc, prec, rec, f1
    ))

    return f2, gmean, roc_auc, acc, prec, rec, f1

def get_regression_results(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print('MAE: {:.3f}\nRMSE: {:.3f}\nR2: {:.3f}\n=========================\n'.format(mae, rmse, r2))

    return mae, rmse, r2

def apply_one_hot_encoding(X_cat):
    ''' given the categorical subset of the dataset and list of categorical features, applies OHE '''
    enc = OneHotEncoder(handle_unknown='ignore')
    data = enc.fit_transform(X_cat)
    try:
        X_cat_encoded = pd.DataFrame(data.toarray(), columns=enc.get_feature_names(list(X_cat.columns)), dtype=int)
    except:
        X_cat_encoded = pd.DataFrame(data.toarray(), columns=enc.get_feature_names_out(list(X_cat.columns)), dtype=int)

    return X_cat_encoded, enc


def get_categorical_features(df):
    ''' given a dataset, gets the list of categorical features using vars2types.pkl dictionary '''
    categorical_features = []
    # Load the variable types
    with open('../map_vars_to_var_types/variable_types/vars2types.pkl', 'rb') as f:
        vars2types = pickle.load(f)

    for col in df.columns:
        if col not in vars2types:
            print(f'Column {col} not in vars2types dictionary')
            continue
        elif vars2types[col] == 'categorical':
            categorical_features.append(col)
    return categorical_features


def load_cols(cols_dir):

    with open(os.path.join(cols_dir, 'numeric.p'), 'rb') as handle:
        numeric = pickle.load(handle)

    with open(os.path.join(cols_dir, 'ordinal.p'), 'rb') as handle:
        ordinal = pickle.load(handle)

    with open(os.path.join(cols_dir, 'categorical.p'), 'rb') as handle:
        categorical = pickle.load(handle)

    return numeric, ordinal, categorical


def mkdir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def plot_learning_curve(X_train, y_train, scaler_txt, model, model_name):
    if scaler_txt == 'MinMax':
        scaler = MinMaxScaler()
    elif scaler_txt == 'Z-score':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    pipeline = make_pipeline(scaler, model)
    #
    # Use learning curve to get training and test scores along with train sizes
    #
    n_jobs = multiprocessing.cpu_count() - 1
    cv = StratifiedKFold()
    train_sizes, train_scores, test_scores = learning_curve(estimator=pipeline, X=X_train, y=y_train,
                                                            cv=cv, train_sizes=np.linspace(0.1, 1.0, 10),
                                                            n_jobs=n_jobs)
    #
    # Calculate training and test mean and std
    #

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    #
    # Plot the learning curve
    #
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.title('Learning Curve')
    plt.xlabel('Training Data Size')
    plt.ylabel('Model accuracy')
    plt.grid()
    plt.legend(loc='lower right')
    # plt.show()
    plt.savefig(os.path.join(save_dir_plots, model_name + '.png'), dpi=300)
    plt.close()


def plot_roc_auc_curve(y_test, y_proba, fig_name, save_dir):
    """ Create ROC AUC plot """
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)

    # Compute AUC
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line for random guess
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    mkdir(save_dir)

    plt.savefig(os.path.join(save_dir, fig_name), dpi=300)
    plt.close()


def plot_precision_recall_curve(y_test, y_proba, save_dir, fig_name):
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_proba)
    # plot the precision-recall curves
    no_skill = len(y_test[y_test == 1]) / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    mkdir(save_dir)

    plt.savefig(os.path.join(save_dir, fig_name + ".png"), dpi=300)
    plt.close()


if __name__ == '__main__':

    df_train = pd.read_csv("../../input/pooled_data_final_train.csv")
    df_test = pd.read_csv("../../input/pooled_data_final_test.csv")

    save_dir = 'results/trained_models/'     # save the trained model
    mkdir(save_dir)

    save_dir_predictions = 'results/actual_vs_predicted/'  # save the actual vs. predicted
    mkdir(save_dir_predictions)

    save_dir_plots = 'results/learning_curves/' # save the learning curves
    mkdir(save_dir_plots)

    save_dir_rocaucplots = "results/roc_curves/"
    mkdir(save_dir_rocaucplots)

    numeric, ordinal, categorical = load_cols(cols_dir="../../input/columns/")

    df_results = pd.DataFrame(columns=['model', 'F2', 'G-mean', 'ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1'])
    df_results_cv = pd.DataFrame(columns=['model', 'F2', 'G-mean', 'ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1'])

    print(f'df_train.shape: {df_train.shape}, df_test.shape: {df_test.shape}')

    X_train, y_train = df_train.loc[:, df_train.columns != "dem1066"], df_train.loc[:, df_train.columns == "dem1066"]
    X_test, y_test = df_test.loc[:, df_test.columns != "dem1066"], df_test.loc[:, df_test.columns == "dem1066"]

    if categorical:
        # X_train_num = X_train[[col for col in X_train.columns if col not in categorical]].reset_index(drop=True)  # numeric X_train
        # X_test_num = X_test[[col for col in X_test.columns if col not in categorical]].reset_index(drop=True)  # numeric X_test

        X_train_num = X_train[[col for col in X_train.columns if col in numeric + ordinal]].reset_index(drop=True)  # numeric X_train
        X_test_num = X_test[[col for col in X_test.columns if col in numeric + ordinal]].reset_index(drop=True)  # numeric X_test

        print(X_train_num.shape, X_test_num.shape)

        X_train_cat = X_train[categorical]  # categorical X_train
        X_test_cat = X_test[categorical]  # categorical X_test

        print(X_train_cat.shape, X_test_cat.shape)

        X_train_ohe, cat_encoder = apply_one_hot_encoding(X_train_cat)  # One-Hot encode the categorical part of X_train
        try:
            X_test_ohe = pd.DataFrame(cat_encoder.transform(X_test_cat).toarray(), columns=cat_encoder.get_feature_names(list(X_test_cat.columns)), dtype=int)  # One-Hot encode the categorical part of X_test
        except:
            X_test_ohe = pd.DataFrame(cat_encoder.transform(X_test_cat).toarray(), columns=cat_encoder.get_feature_names_out(list(X_test_cat.columns)), dtype=int)  # One-Hot encode the categorical part of X_test

        print(X_test_ohe.shape)

        # concatenate numeric + OHE (categorical) X_train , and X_test
        X_train_concat = pd.concat([X_train_num, X_train_ohe], axis=1, ignore_index=True)
        X_train_concat.columns = list(X_train_num.columns) + list(X_train_ohe.columns)
        X_test_concat = pd.concat([X_test_num, X_test_ohe], axis=1, ignore_index=True)
        X_test_concat.columns = list(X_test_num.columns) + list(X_test_ohe.columns)

        print(X_train_concat.shape, X_test_concat.shape)
    else:
        X_train_concat = X_train
        X_test_concat = X_test

    for scaler_txt in ['MinMax']:
        if scaler_txt == 'MinMax':
            scaler = MinMaxScaler()
        elif scaler_txt == 'Z-score':
            scaler = StandardScaler()
        else:
            scaler = RobustScaler()
        for model in models_to_test:
            model_name = models_to_test[model]
            print( f'Cross Validation on MODEL: {model_name} and {scaler_txt} =====================================================================================')
            steps = []
            steps.append(('scaler', scaler))
            steps.append(('model', model()))
            param_g = param_grid[model_name]
            pipeline = Pipeline(steps=steps)

            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
            grid = GridSearchCV(estimator=pipeline, param_grid=param_g, n_jobs=7, cv=cv, scoring='roc_auc', verbose=1)
            # grid = RandomizedSearchCV(pipeline, param_g, verbose=1)
            grid_result = grid.fit(X_train_concat, y_train)
            print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("%f (%f) with: %r" % (mean, stdev, param))
            print('~~~~~~~~~~~~~~ Finished cross validation with hyper parameter tuning ... now Train/Test normally')
            print('TESTING ==============================================================================')
            best_params = grid_result.best_params_
            model_params = {key.split('__')[1]: value for key, value in best_params.items() if key.startswith('model__')}

            optimised_model = model(**model_params)

            print('Plotting the learning curve .....')
            plot_learning_curve(X_train, y_train, scaler_txt, optimised_model, model_name)
            
            # get cross val scores after hyperparameter tuning
            scoring = {
                'accuracy': 'accuracy',
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'f1': make_scorer(fbeta_score, beta=1),
                'f2': make_scorer(fbeta_score, beta=2),
                'gmean': make_scorer(geometric_mean_score),
                'roc_auc': make_scorer(roc_auc_score)
            }

            steps = []
            steps.append(('scaler', scaler))
            steps.append(('model', model(**model_params)))
            param_g = param_grid[model_name]
            pipeline = Pipeline(steps=steps)
            scores = cross_validate(estimator=pipeline, X=X_train_concat, y=y_train, cv=cv, scoring=scoring, n_jobs=-1)
            df_results_cv = pd.concat([df_results_cv,
                                    pd.DataFrame({
                                        'model': [model_name],
                                        'scaler': [scaler_txt],
                                        'F2': [f"{scores['test_f2'].mean():.4f} ± {scores['test_f2'].std():.4f}"],
                                        'G-mean': [f"{scores['test_gmean'].mean():.4f} ± {scores['test_gmean'].std():.4f}"],
                                        'ROC AUC': [f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}"],
                                        'Accuracy': [f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}"],
                                        'Precision': [f"{scores['test_precision'].mean():.4f} ± {scores['test_precision'].std():.4f}"],
                                        'Recall': [f"{scores['test_recall'].mean():.4f} ± {scores['test_recall'].std():.4f}"],
                                        'F1': [f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}"]
                                    })], ignore_index=True)

            df_results_cv.to_excel('results_shallow_cv.xlsx', index=False)

            X_train_scaled = scaler.fit_transform(X_train_concat)
            X_test_scaled = scaler.transform(X_test_concat)

            optimised_model.fit(X_train_scaled, y_train)
            y_pred = optimised_model.predict(X_test_scaled)

            try:
                y_proba = optimised_model.predict_proba(X_test_scaled)[:, 1]  # Select probabilities for the positive class
                plot_roc_auc_curve(y_test=y_test, y_proba=y_proba, fig_name=f"{model_name}_{scaler_txt}", save_dir=save_dir_rocaucplots)
            except:
                y_proba = None
                pass

            with open(os.path.join(save_dir, '{}_{}.pkl'.format(model_name, scaler_txt)), 'wb') as f:
                pickle.dump(optimised_model, f)

            df_actual_predicted = pd.DataFrame()
            df_actual_predicted['actual'] = y_test
            df_actual_predicted['predicted'] = y_pred
            if y_proba is not None:
                df_actual_predicted['probability'] = y_proba
            df_actual_predicted.to_csv(os.path.join(save_dir_predictions, '{}_{}.csv'.format(model_name, scaler_txt)))

            f2, gmean, roc_auc, acc, prec, rec, f1 = get_results(y_test=y_test, y_pred=y_pred)
            df_results = pd.concat([df_results,
                                    pd.DataFrame({
                                        'model': [model_name],
                                        'scaler': [scaler_txt],
                                        'F2': [f2],
                                        'G-mean': [gmean],
                                        'ROC AUC': [roc_auc],
                                        'Accuracy': [acc],
                                        'Precision': [prec],
                                        'Recall': [rec],
                                        'F1': [f1]
                                    })], ignore_index=True)

            df_results = df_results.sort_values(by='F2', ascending=False)
            df_results.to_excel('results_shallow.xlsx', index=False)

            if y_proba is not None:
                sd = "results/precision_recall_curves/"
                mkdir(sd)
                plot_precision_recall_curve(y_test, y_proba, sd, model_name)

        df_results = df_results.sort_values(by='F2', ascending=False)
        df_results.to_excel('results_shallow.xlsx', index=False)
