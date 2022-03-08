import imp
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error,\
                            mean_absolute_error, accuracy_score
from histogram_matching import *

def match_to_normal_distribution(src_dist, mapping=None):

    new_mapping = mapping

    # try:
    if mapping is None:
        # get mean and std
        src_m, src_s = src_dist.mean(), src_dist.std()

        # generate a normal distribution vector with m and s
        dst_dist = np.random.normal(src_m, src_s, src_dist.size)
        dst_dist = np.array(dst_dist, dtype=np.float)

        new_mapping = get_histogram_mapping(source=src_dist, template=dst_dist)

    matched_dist = apply_histogram_mapping(src_dist, new_mapping)

    if mapping is None:
        return matched_dist, new_mapping
    return matched_dist


def get_numeric_categorical_columns(df, target_columns, threshold_prec=5):
    numeric_columns = []
    categorical_columns = []

    for column in df.columns:
        categories = len(set(df[column]))
        prec_categories = (categories / len(df[column])) * 100

        if column not in target_columns:
            print(f'column: {column}, %: {prec_categories}')
            if prec_categories <= threshold_prec:
                categorical_columns.append(column)
            else:
                numeric_columns.append(column)

    print(f'numeric_columns = {numeric_columns}')
    print(f'categorical_columns = {categorical_columns}')

    return numeric_columns, categorical_columns


def transform_dataset(src_df, transformation=match_to_normal_distribution, mapping=None, correction_threshold=0.95, correction=None):
    new_df = pd.DataFrame(index=src_df.index)
    new_mapping = dict() if mapping is None else mapping

    for column in src_df:
        dist = src_df[column].to_numpy()

        # If no mapping was provided
        if mapping is None:
            transformed_dist, new_column_mapping = transformation(dist)
            new_mapping[column] = new_column_mapping
        else:
            transformed_dist = transformation(dist, mapping[column])

        if correction:
            print(f'Correction: {column}')
            transformed_dist = correction(transformed_dist, dist)
            new_mapping[column] = correction(new_mapping[column], dist)

        new_df[column] = transformed_dist

    return new_df, new_mapping


def compare_one_norm_column(regular_train_X, regular_test_X, regular_train_y,
                            regular_test_y, matched_train_X, matched_test_X,
                            yeo_train_X, yeo_test_X, target_column, numerical_columns,
                            acc=False, **kwargs):
    if not acc:
        model = LinearRegression()
    else:
        if 'multi_class' in kwargs:
            model = LogisticRegression(
                multi_class=kwargs['multi_class'], solver=kwargs['solver'])
        else:
            model = LogisticRegression()

    prediction = model.fit(
        regular_train_X, regular_train_y).predict(regular_test_X)
    print('=== Measure for regular model: ===')

    if not acc:
        print(
            f'column={target_column}, r2={r2_score(regular_test_y,prediction)}')
    else:
        print(
            f'column={target_column}, acc={accuracy_score(regular_test_y,prediction)}')

    print("Mean Absolute Perc Error (Σ(|y - pred|/y)/n):",
          "{:,.3f}".format(mean_absolute_percentage_error(regular_test_y, prediction)))
    print("Mean Absolute Error (Σ|y - pred|/n):",
          "{:,.0f}".format(mean_absolute_error(regular_test_y, prediction)))
    print("Root Mean Squared Error (sqrt(Σ(y - pred)^2/n)):",
          "{:,.0f}".format(np.sqrt(mean_squared_error(regular_test_y, prediction))))
    print('=' * 20)

    for norm_column in numerical_columns:
        print(f'\n=== Compare for column {norm_column} ===')
        norm_df_train_X = regular_train_X.drop(norm_column, axis=1)
        norm_df_train_X[norm_column] = matched_train_X[norm_column]

        norm_df_test_X = regular_test_X.drop(norm_column, axis=1)
        norm_df_test_X[norm_column] = matched_test_X[norm_column]

        yeo_norm_train_X = regular_train_X.drop(norm_column, axis=1)
        yeo_norm_train_X[norm_column] = yeo_train_X[norm_column]

        yeo_norm_test_X = regular_test_X.drop(norm_column, axis=1)
        yeo_norm_test_X[norm_column] = yeo_test_X[norm_column]

        if not acc:
            matched_model = LinearRegression()
            yeo_model = LinearRegression()
        else:
            if 'multi_class' in kwargs:
                matched_model = LogisticRegression(
                    multi_class=kwargs['multi_class'], solver=kwargs['solver'])
                yeo_model = LogisticRegression(
                    multi_class=kwargs['multi_class'], solver=kwargs['solver'])
            else:
                matched_model = LogisticRegression()
                yeo_model = LogisticRegression()

        matched_prediction = matched_model.fit(
            norm_df_train_X, regular_train_y).predict(norm_df_test_X)
        yeo_prediction = yeo_model.fit(
            yeo_norm_train_X, regular_train_y).predict(yeo_norm_test_X)

        if not acc:
            print(
                f'column={norm_column}, matched r2={r2_score(regular_test_y,matched_prediction)}')
            print(
                f'column={norm_column}, yeo r2={r2_score(regular_test_y,yeo_prediction)}')
        else:
            print(
                f'column={norm_column}, matched acc={accuracy_score(regular_test_y,matched_prediction)}')
            print(
                f'column={norm_column}, yeo acc={accuracy_score(regular_test_y,yeo_prediction)}')

    print('\n' + '=' * 50)
