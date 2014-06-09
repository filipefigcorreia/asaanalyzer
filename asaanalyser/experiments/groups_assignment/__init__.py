# -*- coding: utf-8 -*-
import os
import csv
from random import random
import numpy as np
from scipy import stats
from asaanalyser.common.util import CSVUnicodeWriter
from asaanalyser.common.stats import get_simple_stats, get_mww

def make_assignments(students, grades, output_dir, username_prefix = "p"):
    # Beware: This line discards courses with no grade. This means the averages
    # of each student may be computed using different sets of courses.
    grades = [[gr[0]] + [int(float(g)) for g in gr[1:] if g] for gr in grades]


    assignment_cache_filename = os.path.join(output_dir, 'assignment.csv')
    stats_filename = os.path.join(output_dir, 'stats.csv')
    tex_control_stats_filename = os.path.join(output_dir, 'tex/control.tex')
    tex_experimental_stats_filename = os.path.join(output_dir, 'tex/experimental.tex')
    tex_shapiro_filename = os.path.join(output_dir, 'tex/shapiro.tex')
    tex_levene_filename = os.path.join(output_dir, 'tex/levene.tex')
    tex_ttest_filename = os.path.join(output_dir, 'tex/ttest.tex')
    tex_mwu_filename = os.path.join(output_dir, 'tex/mwu.tex')


    ################################
    # make assignment.
    # save results file to csv and always load from there on subsequent runs if
    # the file exists. This allows to keep a record of the assignment, as well as
    # to compute the statistical tests without loosing the previous assignment
    try:
        with open(assignment_cache_filename, 'rb') as csvfile:
            csvreader = csv.reader(csvfile) # [group, username, student_number],
            csvreader.next() # skip header
            data = list(csvreader)
            students_cg = [row for row in data if row[0] == "control_group"]
            students_eg = [row for row in data if row[0] == "experimental_group"]
    except IOError as e:
        usernames_cg = ["%s%03d" % (username_prefix,i) for i in range(1, 61)] # hardcoded limit to 60 usernames
        usernames_eg = ["%s%03d" % (username_prefix,i) for i in range(61, 121)] # hardcoded limit to 60 usernames

        randomized_students = sorted(students, key=lambda x: random())
        randomized_students_nrs = [student[1] for student in randomized_students]

        def assign(usernames, randomized_student_nrs, group):
            return zip(
                [group]*len(usernames),
                usernames,
                randomized_student_nrs
            ) # [group, username, student_number]

        students_cg = assign(usernames_cg, randomized_students_nrs[0:len(randomized_students_nrs)/2], "control_group")
        students_eg = assign(usernames_eg, randomized_students_nrs[len(randomized_students_nrs)/2:], "experimental_group")

        writer = CSVUnicodeWriter(open(assignment_cache_filename, "wb"))
        writer.writerow(['group', 'username', 'student_number'])
        writer.writerows(students_cg)
        writer.writerows(students_eg)

    print "Control Group Students:", students_cg
    print "Experimental Group Students:", students_eg

    #### splitting average grades into the two different groups

    avg_grades = [(gr[0], np.average(gr[1:])) for gr in grades] # [student_nr, avg]
    avg_grades_cg = [g[1] for g in avg_grades if g[0] in np.array(students_cg)[:,2]]
    avg_grades_eg = [g[1] for g in avg_grades if g[0] in np.array(students_eg)[:,2]]

    print "Control Group Grades:", avg_grades_cg
    print "Experimental Group Grades:", avg_grades_eg


    #### descriptive statistics
    cg_sstats = ("control_group",) + get_simple_stats(avg_grades_cg) 
    eg_sstats = ("experimental_group",) + get_simple_stats(avg_grades_eg) 
    writer = CSVUnicodeWriter(open(stats_filename, "wb"))
    writer.writerow(['group', 'sum', 'avg', 'mean', 'median', 'std', 'var', 'count']) #, 'sem'])
    writer.writerow(cg_sstats)
    writer.writerow(eg_sstats)

    def write_sstats_tex_file(sstats, filename):
        tex_file = open(filename, "w")
        tex_file.write("{0} & {1:.2f} & {2:.4f}".format(sstats[7], sstats[3], sstats[5]))
        tex_file.close()
    write_sstats_tex_file(cg_sstats, tex_control_stats_filename)
    write_sstats_tex_file(eg_sstats, tex_experimental_stats_filename)


    #### test for normal distribution (required assumption for the ttest)
    
    # using the Shapiro-Wilk test
    from scipy.stats import shapiro
    sw_w, sw_p_value = shapiro(avg_grades_cg + avg_grades_eg)
    print "\nResult of the Shapiro-Wilk test for normality. w: {0}; p_value: {1}".format(sw_w, sw_p_value)
    if sw_p_value >= 0.05:
        print "%f >= 0.05, so we can assume a normal distribution." % (sw_p_value,)
    else:
        print "%f < 0.05, so we cannot assume a normal distribution." % (sw_p_value,)


    ##### test for group equivalence using a ttest

    from asaanalyser.common.stats import get_ttest_equal_var, get_ttest_diff_var, get_levene

    lev_p_value, lev_w = get_levene(avg_grades_cg, avg_grades_eg)
    print "\nResult of the Levene test. Sig: %f, W: %f" % (lev_p_value, lev_w)
    eqvar_statistic, eqvar_p_value, _, _, _, _ = get_ttest_equal_var(avg_grades_cg, avg_grades_eg)
    dfvar_statistic, dfvar_p_value, _, _, _, _ = get_ttest_diff_var(avg_grades_cg, avg_grades_eg)
    if lev_p_value >= 0.05:
        print "%f >= 0.05, so we can assume an equal variance" % (lev_p_value, )
        p_value, statistic = eqvar_p_value, eqvar_statistic
    else:
        print "%f < 0.05, so we can assume a diff variance" % (lev_p_value, )
        p_value, statistic = dfvar_p_value, dfvar_statistic

    print "\nResult of the ttest. Sig: %f, t-statistic: %f" % (p_value, statistic)
    if p_value >= 0.05:
        print "%f >= 0.05, so we can conclude that there is no statistically significant difference between the two groups" % (p_value, )
    else:
        print "%f < 0.05, so we cannot conclude that there is no statistically significant difference between the two groups" % (p_value, )



    #### If not dealing with normal distributions, then a Wilcoxon Rank Sum (aka Mann-Whitney U) is a better option

    # Mann-Whitney U
    u, pvalue, _, _ = get_mww(avg_grades_cg, avg_grades_eg)
    if pvalue <= 0.05:
        if u < 0:
            match = "+"
        else:
            match = "-"
    else:
        match = "="
    print "\nResult of the Mann-Whitney U test. u: {0}; p: {1}".format(u, pvalue)
    print match



    # output the remaining tex files
    tex_file = open(tex_shapiro_filename, "w")
    tex_file.write("""{0:.3f} & {1:.3f}""".format(sw_w, sw_p_value))
    tex_file.close()

    tex_file = open(tex_levene_filename, "w")
    tex_file.write("""{0:.3f} & {1:.3f}""".format(lev_w, lev_p_value))
    tex_file.close()

    tex_file = open(tex_ttest_filename, "w")
    tex_file.write(""" $\\neq$ & {0:.3f} & {1:.3f} \\\\""".format(eqvar_statistic, eqvar_p_value))
    tex_file.close()

    tex_file = open(tex_mwu_filename, "w")
    tex_file.write(""" $\\neq$ & {0:.3f} & {1:.3f} \\\\""".format(u, pvalue))
    tex_file.close()


    #### Draw box-and-whiskers for grades. One plot per group and subject
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.direction'] = 'out'

    fig = plt.figure(figsize=(5, 6), dpi=80)
    xlabels = ["CG", "EG"]
    groups_values = [avg_grades_cg, avg_grades_eg]
    ax = fig.add_subplot(111)
    ax.set_aspect(0.2)
    ax.boxplot(groups_values, widths=0.5)
    ax.set_xticks(np.arange(len(xlabels))+1)
    ax.set_xticklabels(xlabels)
    fig.subplots_adjust(bottom=0.3)

    yticks = np.linspace(0, 20, 5)
    ylabels = ["{0:.0f}".format(ylabel) for ylabel in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)

    plt.savefig(os.path.join(output_dir, "groups.pdf"))