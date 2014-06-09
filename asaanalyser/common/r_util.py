# -*- coding: utf-8 -*-
# Credits:
# Chris Rodgers - https://github.com/cxrodgers/my/blob/master/stats.py
# Brent Pedersen - https://gist.github.com/brentp/853885
# Filipe Correia


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The next block was borrowed and adapted from Chris Rodgers
# https://github.com/cxrodgers/my/blob/master/stats.py

import rpy2.robjects as robjects
r = robjects.r

def check_float_conversion(a1, a2, tol):
    """Checks that conversion to R maintained uniqueness of arrays.

    a1 : array of unique values, typically originating in Python
    a2 : array of unique values, typically grabbed from R

    If the lengths are different, or if either contains values that
    are closer than `tol`, an error is raised.
    """
    if len(a1) != len(a2):
        raise ValueError("uniqueness violated in conversion")
    if len(a1) > 1:
        if np.min(np.diff(np.sort(a1))) < tol:
            raise ValueError("floats separated by less than tol")
        if np.min(np.diff(np.sort(a2))) < tol:
            raise ValueError("floats separated by less than tol")

def r_utest(x, y, alternative='two.sided', mu=0, verbose=False, tol=1e-6, exact='FALSE',
    fix_nan=True, fix_float=False, paired='FALSE'):
    """Mann-Whitney U-test in R

    This is a test on the median of the distribution of sample in x minus
    sample in y. It uses the R implementation to avoid some bugs and gotchas
    in scipy.stats.mannwhitneyu.

    Some care is taken when converting floats to ensure that uniqueness of
    the datapoints is conserved, which should maintain the ranking.

    x : dataset 1
    y : dataset 2
        If either x or y is empty, prints a warning and returns some
        values that indicate no significant difference. But note that
        the test is really not appropriate in this case.
    alternative : a character string specifying the alternative hypothesis,
        must be one of "two.sided" (default), "greater" or "less". You can
        specify just the initial letter.
    mu : null hypothesis on median of sample in x minus sample in y
    verbose : print a bunch of output from R
    tol : if any datapoints are closer than this, raise an error, on the
        assumption that they are only that close due to numerical
        instability
    exact : see R doc
        Defaults to FALSE since if the data contain ties and exact is TRUE,
        R will print a warning and approximate anyway
    fix_nan : if p-value is nan due to all values being equal, then
        set p-value to 1.0. But note that the test is really not appropriate
        in this case.
    fix_float : int, or False
        if False or if the data is integer, does nothing
        if int and the data is float, then the data are multiplied by this
        and then rounded to integers. The purpose is to prevent the numerical
        errors that are tested for in this function. Note that differences
        less than 1/fix_float will be removed.

    Returns: dict with keys ['U', 'p', 'auroc']
        U : U-statistic.
            Large U means that x > y, small U means that y < x
            Compare scipy.stats.mannwhitneyu which always returns minimum U
        p : two-sided p-value
        auroc : area under the ROC curve, calculated as U/(n1*n2)
            Values greater than 0.5 indicate x > y
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # What type of R object to create
    if x.dtype.kind in 'iu' and y.dtype.kind in 'iu':
        behavior = 'integer'
    elif x.dtype.kind == 'f' or y.dtype.kind == 'f':
        behavior = 'float'
    else:
        raise ValueError("cannot determine datatype of x and y")

    # Optionally fix float
    if fix_float and behavior == 'float':
        x = np.rint(x * fix_float).astype(np.int)
        y = np.rint(y * fix_float).astype(np.int)
        behavior = 'integer'

    # Define variables
    if behavior == 'integer':
        robjects.globalenv['x'] = robjects.IntVector(x)
        robjects.globalenv['y'] = robjects.IntVector(y)
    elif behavior == 'float':
        robjects.globalenv['x'] = robjects.FloatVector(x)
        robjects.globalenv['y'] = robjects.FloatVector(y)

        # Check that uniqueness is maintained
        ux_r, ux_p = r("unique(x)"), np.unique(x)
        check_float_conversion(ux_r, ux_p, tol)
        uy_r, uy_p = r("unique(y)"), np.unique(y)
        check_float_conversion(uy_r, uy_p, tol)

        # and of the concatenated
        uxy_r, uxy_p = r("unique(c(x,y))"), np.unique(np.concatenate([x,y]))
        check_float_conversion(uxy_r, uxy_p, tol)

    # Run the test
    if len(x) == 0 or len(y) == 0:
        print "warning empty data in utest, returning p = 1.0"
        U, p, auroc = 0.0, 1.0, 0.5
    else:
        res = r("wilcox.test(x, y, alternative=\"%s\", mu=%r, exact=%s, paired=%s)" % (alternative, mu, exact, paired))
        U, p = res[0][0], res[2][0]
        auroc = float(U) / (len(x) * len(y))

    # Fix p-value
    if fix_nan and np.isnan(p):
        p = 1.0

    # debug
    if verbose:
        print behavior
        s_x = str(robjects.globalenv['x'])
        print s_x[:1000] + '...'
        s_y = str(robjects.globalenv['y'])
        print s_y[:1000] + '...'
        print res

    return {'U': U, 'p': p, 'auroc': auroc}



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# The next block was borrowed from Brent Pedersen
# https://gist.github.com/brentp/853885

import numpy as np
from rpy2.robjects.packages import importr
R = robjects.r

def rhelp(fn_name, utils=importr("utils")):
    str(utils.help(fn_name))

def pify(rthing):
    """
    turn an r thing into a python thing
    >>> pify(R("2 * 2"))
    4.0

    >>> pify(R("c(1, 2, 3)"))
    [1.0, 2.0, 3.0]

    >>> pify(R("t.test(1:4, 1:4)"))
    {'null.value': {'difference in means': 0.0}, 'data.name': '1:4 and 1:4', 'method': 'Welch Two Sample t-test', 'p.value': 1.0, 'statistic': {'t': 0.0}, 'estimate': {'mean of y': 2.5, 'mean of x': 2.5}, 'conf.int': [-2.2337146951647044, 2.2337146951647044], 'parameter': {'df': 5.9999999999999982}, 'alternative': 'two.sided'}


    >>> a = np.arange(10)
    >>> b = np.array([2, 12, 4, 6, 1, 8, 9, 1, 3, 1])
    >>> ttest = R['t.test']

    >>> pify(ttest(a, b, alternative="two.sided"))["p.value"]
    0.89939605650576726

    >>> pify(ttest(a, b, alternative="less"))["p.value"]
    0.44969802825288363

    >>> chisquare = R['chisq.test']
    >>> A = [122, 14, 28, 11]
    >>> kwargs = {'simulate.p.value':True}
    >>> pify(chisquare(robjects.IntVector(A)))
    {'observed': [122, 14, 28, 11], 'residuals': [11.830288005188812, -4.4977772288098041, -2.3811761799581315, -4.9513345964208764], 'p.value': 5.0742757901326037e-41, 'statistic': {'X-squared': 190.37142857142857}, 'expected': [43.75, 43.75, 43.75, 43.75], 'data.name': 'c(122L, 14L, 28L, 11L)', 'parameter': {'df': 3.0}, 'method': 'Chi-squared test for given probabilities'}

    >>> df = R('data.frame(acol=1:4, bcol=letters[1:4])')
    >>> pify(df)
    rec.array([(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')], 
          dtype=[('acol', '<i4'), ('bcol', '|S1')])


    """
    if isinstance(rthing, robjects.vectors.DataFrame):
        _r_unfactor(rthing)
        return np.rec.fromarrays(rthing, names=tuple(rthing.colnames))
    if hasattr(rthing, "nrow"):
        m = np.array(list(rthing)).reshape(rthing.nrow, rthing.ncol)
        return m

    if not hasattr(rthing, "iteritems"): 
        return rthing
    d = {}
    l = []
    for k, v in rthing.iteritems():
        if k is None:
            l.append(pify(v))
        else:
            d[k] = pify(v)
    if d and len(d) == 1 and None in d:
        return d[None]
    if l and len(l) == 1:
        # could be a list of length 1, but cant tell...
        return l[0]
    return d or l

def _r_unfactor(rdf):
    """
    convert factor vectors back to string
    """
    for i, col in enumerate(rdf.colnames):
        if r['is.factor'](rdf[i])[0]:
            rdf[i] = r['as.character'](rdf[i])


if __name__ == "__main__":

    import doctest
    doctest.testmod(verbose=0)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from rpy2.robjects.vectors import StrVector
stats = importr("stats")

def r_ttest(x, y, alternative='two.sided', equal_variance=False, paired=False):
    result = pify(stats.t_test(robjects.FloatVector(x), robjects.FloatVector(y), **{'alternative': StrVector((alternative, )), 'var.equal': equal_variance, 'paired': paired}))
    return {'p': result['p.value'], 't': result['statistic']['t']}
