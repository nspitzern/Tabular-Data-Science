# From: https://stackoverflow.com/a/45091961/11915280
from scipy.stats import ks_2samp
from scipy.special import kl_div
from math import e


def get_entropy(df, base=None):
    ent = []
    for column in df.columns:
        value, counts = np.unique(df[column], return_counts=True)
        norm_counts = counts / counts.sum()
        base = e if base is None else base
        ent.append(-(norm_counts * np.log(norm_counts)/np.log(base)).sum())
    return np.array(ent)


def get_insights(df):
    return pd.DataFrame([i for i in zip(df.min(), df.max(), df.mean(), df.median(), df.std(), df.var(), df.skew(), df.kurtosis(), get_entropy(df))],
                        index=df.columns,
                        columns=['min', 'max', 'mean', 'median', 'std', 'var', 'skew', 'kurtosis', 'entropy'])


# def kl_divergence(p, q):
#     return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def kl_divergence(df_p, df_q):
    dkl = []
    for column in df_p.columns:
        p_values, p_counts = np.unique(df_p[column], return_counts=True)
        q_values, q_counts = np.unique(df_q[column], return_counts=True)
        p = p_counts / p_counts.sum()
        q = q_counts / q_counts.sum()

        p_probs = np.zeros(max(p, q))
        q_probs = np.zeros(max(p, q))

        p_probs[p_values] = p_counts
        q_probs[q_values] = q_counts

        kl = kl_div(p, q)
        dkl.append(sum(kl))
        # dkl.append(np.sum(np.where(p != 0, p * np.log(p / q), 0)))
    return np.array(dkl)


def find_data_distribution(column_name: str, column):
    mean, std, size = column.mean(), column.std(), len(column)
    min, max = column.min(), column.max()
    dists = [rnd.normal, rnd.power, rnd.lognormal, rnd.poisson,
             rnd.weibull, rnd.uniform, rnd.pareto, rnd.standard_t, rnd.dirichlet]

    for dist in dists:
        # simple example - from uniform to normal
        target_dist = np.random.uniform(min, max, size)
        if dist is not np.random.uniform:
            target_dist = dist(target_dist)

        # statistic, p_value = ks_2samp(column, target_dist)

        # x = np.arange(min, max, float(min + max) / size)

        # target_dist = np.where(target_dist == np.NAN or target_dist == np.inf, self.clip_value, target_dist)

        combined_df = pd.DataFrame({
            f'{column_name}': column,
            f'{dist.__name__}': target_dist
        })
        # combined_df.plot(title=f'KL({column_name}||{dist.__name__})) = %1.3f' % kl_divergence(column, target_dist), kind='density')

        plot_numerical_histograms(combined_df, [f'{column_name}'])

        # column.plot.kde()

        # plt.scatter(column, target_dist)
        # plt.show()

        # plt.title(f'KL({column_name}||{dist.__name__})) = %1.3f' % kl_divergence(column, target_dist))
        # plt.plot(x, column)
        # plt.plot(x, target_dist, c='red')
