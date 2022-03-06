from scipy import stats
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt


def plot_histograms_side_by_side(df1, df2):
    cols = 2
    rows = df1.shape[1]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    fig.subplots_adjust(hspace=0.5)

    for row, column in enumerate(df1.columns):
        # get the p_value of the KS test for the current column
        m1, s1 = stats._continuous_distns.norm.fit(df1[column])
        ks_res1, ks_pval1 = scipy.stats.kstest(
            df1[column].values, stats._continuous_distns.norm.cdf, args=(m1, s1))
        shapiro_res1, shapiro_pval1 = stats.shapiro(df1[column].values)

        # build the histograms
        df1[column].hist(ax=axes[row, 0])
        axes[row, 0].set_title(
            f'{column}\n KS p_val={ks_pval1:.5e}\n Shapiro p_val={shapiro_pval1:.5e}')

        # get the p_value of the KS test for the current column
        m2, s2 = stats._continuous_distns.norm.fit(df2[column])
        ks_res2, ks_pval2 = scipy.stats.kstest(
            df2[column].values, stats._continuous_distns.norm.cdf, args=(m2, s2))
        shapiro_res2, shapiro_pval2 = stats.shapiro(df2[column].values)

        # build the histograms
        df2[column].hist(ax=axes[row, 1])
        axes[row, 1].set_title(
            f'{column}\n KS p_val={ks_pval2:.5e}\n Shapiro p_val={shapiro_pval2}')


def plot_numerical_histograms(df, numeric_columns):
    # we will create a histogram for each categorical attribute
    n = len(numeric_columns)
    cols = 3
    rows = (n//3) + 1 if n % 3 != 0 else (n // 3)
    #max_bars = 8

    # generate a figures grid:
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(numeric_columns):
        # calculate the current place on the grid
        r = int(i/cols)
        c = i % cols

        # get the p_value of the KS test for the current column
        m, s = stats._continuous_distns.norm.fit(df[column])
        ks_res, ks_pval = scipy.stats.kstest(
            df[column].values, stats._continuous_distns.norm.cdf, args=(m, s))

        shapiro_res, shapiro_pval = stats.shapiro(df[column].values)

        # build the histograms
        df[column].hist(ax=axes[r, c])
        axes[r, c].set_title(
            f'{column}\n KS p_val={ks_pval:.5e}\n Shapiro p_val={shapiro_pval:.5e}')


def qq_plot_data(df, numeric_columns):
    # we will create a histogram for each categorical attribute
    n = len(numeric_columns)
    cols = 3
    rows = (n//3) + 1 if n % 3 != 0 else (n//3)
    #max_bars = 8

    # generate a figures grid:
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(numeric_columns):
        # calculate the current place on the grid
        r = int(i/cols)
        c = i % cols

        m = df[column].mean()
        s = df[column].std()

        # build the qqplots
        sm.qqplot(df[column].dropna(), loc=m, scale=s, line='r', ax=axes[r, c])
        axes[r, c].set_title(column)


def plot_correlation_heatmaps(original_df, matched_df, yeo_df, columns):
    original_df_corr = original_df[columns].corr(method="pearson").loc[columns]
    matched_df_corr = matched_df[columns].corr(method="pearson").loc[columns]
    yeo_df_corr = yeo_df[columns].corr(method="pearson").loc[columns]

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(60, 60))
    sns.heatmap(original_df_corr, annot=True, fmt='.2f',
                cmap="YlGnBu", cbar=True, linewidths=0.5, ax=ax1)
    ax1.set_title('Original')
    sns.heatmap(matched_df_corr, annot=True, fmt='.2f',
                cmap="YlGnBu", cbar=True, linewidths=0.5, ax=ax2)
    ax2.set_title('Matched')
    sns.heatmap(yeo_df_corr, annot=True, fmt='.2f',
                cmap="YlGnBu", cbar=True, linewidths=0.5, ax=ax3)
    ax3.set_title('Yeo-Johnson')
    sns.heatmap(original_df_corr - matched_df_corr, annot=True,
                fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5, ax=ax4)
    ax4.set_title('Original-Matched Difference')
    sns.heatmap(original_df_corr - yeo_df_corr, annot=True,
                fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5, ax=ax5)
    ax5.set_title('Original-Yeo Difference')


def plot_histograms(original, new, yeo):
    cols = 3
    n = len(original.columns)
    rows = (n//3) + 1 if n % 3 != 0 else (n // 3)

    # generate a figures grid:
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    fig.subplots_adjust(hspace=0.5)

    for i, (col1, col2, col3) in enumerate(zip(original, new, yeo)):
        # calculate the current place on the grid
        r = int(i/cols)
        c = i % cols

        collective_df = pd.DataFrame({
            'original': original[col1],
            'new': new[col2],
            'yeo': yeo[col3]
        })

        collective_df['original'].hist(alpha=1, ax=axes[r, c], color='red')
        collective_df['new'].hist(alpha=0.3, ax=axes[r, c], color='green')
        collective_df['yeo'].hist(alpha=0.3, ax=axes[r, c], color='blue')

        axes[r, c].set_title(f'{col1}')
        axes[r, c].legend(['original', 'new', 'yeo'])

# def plot_histograms(df, numeric_columns):
#     #we will create a histogram for each categorical attribute
#     n=len(numeric_columns)
#     cols = 3
#     rows=(n//3) + 1 if n%3 !=0 else (n // 3)
#     #max_bars = 8

#     #generate a figures grid:
#     fig, axes = plt.subplots(rows,cols,figsize=(cols*5,rows*5))
#     fig.subplots_adjust(hspace=0.5)

#     for i, column in enumerate(numeric_columns):
#         #calculate the current place on the grid
#         r=int(i/cols)
#         c=i%cols

#         # get the p_value of the KS test for the current column
#         m,s = stats._continuous_distns.norm.fit(df[column])
#         res, pval = scipy.stats.kstest(df[column].values,stats._continuous_distns.norm.cdf,args=(m,s))

#         # build the histograms
#         df[column].hist(ax=axes[r,c])
#         axes[r,c].set_title(f'{column}, p_val={pval:.5e}')
