# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind, levene

def get_mww_deprecated(group1, group2):
    mww_u, mww_p_value = mannwhitneyu(group1, group2)
    return (mww_p_value*2, mww_u)

def get_ttest_equal_var_deprecated(group1, group2):
    try:
        tt_statistic, tt_p_value = ttest_ind(group1, group2)
        return (tt_p_value, tt_statistic)
    except ZeroDivisionError:
        return (0, 0)

def get_ttest_diff_var_deprecated(group1, group2):
    try:
        tt_statistic, tt_p_value = ttest_ind(group1, group2, equal_var = False)
        return (tt_p_value, tt_statistic)
    except ZeroDivisionError:
        return (0, 0)

def get_levene(group1, group2):
    lev_w, lev_p_value = levene(group1, group2)
    return (lev_p_value, lev_w)

def get_shapiro(data):
    from scipy.stats import shapiro
    w, p_value = shapiro(data)
    return (p_value, w)

def get_simple_stats(group):
    return (
        np.sum(group),
        np.average(group),
        np.mean(group),
        np.median(group),
        np.std(group),
        np.var(group),
        np.count_nonzero(group),
    )

from r_util import r_utest, r_ttest

def get_mww(group1, group2):
    mww_twotailed = r_utest(group1, group2)
    mww_lessthan = r_utest(group1, group2, alternative='l')
    mww_greaterthan = r_utest(group1, group2, alternative='g')
    return (mww_twotailed['U'], mww_twotailed['p'], mww_lessthan['p'], mww_greaterthan['p'])

def _get_ttest(group1, group2, equal_variance):
    tt_twotailed = r_ttest(group1, group2)
    tt_lessthan = r_ttest(group1, group2, alternative='l', equal_variance=equal_variance)
    tt_greaterthan = r_ttest(group1, group2, alternative='g')
    return (tt_twotailed['t'], tt_twotailed['p'], tt_lessthan['t'], tt_lessthan['p'], tt_greaterthan['t'], tt_greaterthan['p'])

def get_ttest_equal_var(group1, group2):
    return _get_ttest(group1, group2, True)

def get_ttest_diff_var(group1, group2):
    return _get_ttest(group1, group2, False)

