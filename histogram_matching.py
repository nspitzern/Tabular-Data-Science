import numpy as np

# Histogram matching algorithm (Matlab/Python): https://math-wiki.com/images/e/e9/89-560t2018-10-31.pdf, https://medium.com/hackernoon/histogram-equalization-in-python-from-scratch-ebb9c8aa3f23, https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x


def get_histogram_mapping(source, template):
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values


def apply_histogram_mapping(source, hist_mapping):
    oldshape = source.shape
    source = source.ravel()
    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    bin_idx = bin_idx.clip(0, hist_mapping.size - 1)

    return hist_mapping[bin_idx].reshape(oldshape)


def histogram_matching(source, template):
    mapping = get_histogram_mapping(source, template)
    return apply_histogram_mapping(source, mapping)
