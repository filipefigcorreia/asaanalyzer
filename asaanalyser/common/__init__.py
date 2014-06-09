# -*- coding: utf-8 -*-
import math
import os
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator


from util import CSVUnicodeWriter, xls_to_lists, td_to_str, neighborhood
from model import ASADatabase, FakeASADatabase, ASAExperiment, ASAExperimentGroup, Measurements

def _write_csv_for_asadataset(filename, asa_ds):
    writer = CSVUnicodeWriter(open(filename, "wb"))
    writer.writerow(asa_ds.headers)
    writer.writerows(asa_ds.data)

def print_xls_data_as_csv(output_path_csv, tasks_times):
    # Dump xls data to csvs. One csv per experimental group.
    for grp_name, (grp_headers, grp_data) in tasks_times:
        writer = CSVUnicodeWriter(open(os.path.join(output_path_csv, "00_xls_" + grp_name + ".csv"), "wb"))
        writer.writerow(grp_headers)
        writer.writerows(grp_data)

def print_db_data_as_csv(output_path_csv, experiment):

    # Dump raw db data to a single csv.
    #for db_filename in db_filenames:
    #    ds = dataset_getter(db_filename)
    #    filename_without_extension = path.splitext(path.basename(db_filename))[0]
    #    _write_csv_for_asadataset(output_path_csv + "00_" + filename_without_extension + ".csv", ds)

    # Dump db data to csvs. One csv per experimental group.
    for group in experiment.groups:
        _write_csv_for_asadataset(os.path.join(output_path_csv, "00_db_" + group.name + ".csv"), group.db_dataset)


def print_results(output_path_csv,
                  output_path_tex,
                  output_path_pdf,
                  experiment):

    asa_calcs = experiment.run_tests()

    from sys import stdout
    stdout.write("Creating outputs")
    stdout.flush()

    # Write intermediate and final calculations to csvs
    calculations = asa_calcs.as_tuples_list()
    for i in range(len(calculations)):
        name = calculations[i][0]
        asa_ds = calculations[i][1]
        _write_csv_for_asadataset(os.path.join(output_path_csv, "%02d_%s.csv" % (i+1, name)), asa_ds)

    # Write tex files
    calculations = asa_calcs.as_dict()

    from util import seconds_to_td_str
    def write_file_tex_table(table_data, filename, columns_to_underline=None, underline_lambda=None):
        tex_file = open(os.path.join(output_path_tex, filename), "w")
        for row in table_data:
            for column_idx, value in enumerate(row):
                tex_value = ""
                if type(value) == float:
                    tex_value += "{0:.3f}".format(value)
                else:
                    tex_value += "{0}".format(value)
                if not columns_to_underline is None and column_idx in columns_to_underline and underline_lambda(value):
                    tex_value = "\\underline{{{0}}}".format(tex_value)
                if not column_idx == 0:
                    tex_file.write(" & ")
                tex_file.write(tex_value)
            tex_file.write("\\\\\n")
        tex_file.close()

    def write_file_tex_list(data, filename):
        tex_file = open(os.path.join(output_path_tex, filename), "w")
        is_first_in_row = True
        for row in data:
            if not is_first_in_row:
                tex_file.write("\\\\\n")
            tex_file.write("\\noindent \\textbf{{{0}.}} {1}".format(*row))
            is_first_in_row = False
        tex_file.close()

    # Descriptive statistics for activity times

    activity_times_sstats_ds = calculations['activity_times_sstats']
    cg_idxs = np.array(activity_times_sstats_ds.data)[:,0] == experiment.groups[0].name
    eg_idxs = np.array(activity_times_sstats_ds.data)[:,0] == experiment.groups[1].name
    cg_data = np.array(activity_times_sstats_ds.data)[cg_idxs][:,1:].tolist()
    eg_data = np.array(activity_times_sstats_ds.data)[eg_idxs][:,1:].tolist()
    write_file_tex_table(
        [
            [
                "",
                Measurements.name_to_pretty_name(row[0]),
                seconds_to_td_str(float(row[1]), use_ms=False),
                seconds_to_td_str(float(row[3]), use_ms=False),
                seconds_to_td_str(float(row[5]), use_ms=False)
            ] for row in cg_data
        ],
        'activ_ds_cg.tex')
    write_file_tex_table(
        [
            [
                "",
                Measurements.name_to_pretty_name(row[0]),
                seconds_to_td_str(float(row[1]), use_ms=False),
                seconds_to_td_str(float(row[3]), use_ms=False),
                seconds_to_td_str(float(row[5]), use_ms=False)
            ] for row in eg_data
        ],
        'activ_ds_eg.tex')

    # Descriptive statistics for activity times aggregated by issue

    issues_activity_times_sstats_ds = calculations['issues_activity_times_sstats']
    cg_idxs = np.array(issues_activity_times_sstats_ds.data)[:,0] == experiment.groups[0].name
    eg_idxs = np.array(issues_activity_times_sstats_ds.data)[:,0] == experiment.groups[1].name
    cg_data = np.array(issues_activity_times_sstats_ds.data)[cg_idxs][:,1:].tolist()
    eg_data = np.array(issues_activity_times_sstats_ds.data)[eg_idxs][:,1:].tolist()
    write_file_tex_table(
        [
            [
                "",
                row[0].title(),
                seconds_to_td_str(float(row[1]), use_ms=False),
                seconds_to_td_str(float(row[3]), use_ms=False),
                seconds_to_td_str(float(row[5]), use_ms=False)
            ] for row in cg_data
        ],
        'issues_activ_ds_cg.tex')
    write_file_tex_table(
        [
            [
                "",
                row[0].title(),
                seconds_to_td_str(float(row[1]), use_ms=False),
                seconds_to_td_str(float(row[3]), use_ms=False),
                seconds_to_td_str(float(row[5]), use_ms=False)
            ] for row in eg_data
        ],
        'issues_activ_ds_eg.tex')

    # Descriptive statistics for total activity times

    total_activity_times_sstats_ds = calculations['total_activity_times_sstats']
    cg_data = np.array(total_activity_times_sstats_ds.data)[0].tolist()
    eg_data = np.array(total_activity_times_sstats_ds.data)[1].tolist()
    write_file_tex_table([["", seconds_to_td_str(float(cg_data[1]), use_ms=False), seconds_to_td_str(float(cg_data[3]), use_ms=False), seconds_to_td_str(float(cg_data[5]), use_ms=False)]], 'tactiv_ds_cg.tex')
    write_file_tex_table([["", seconds_to_td_str(float(eg_data[1]), use_ms=False), seconds_to_td_str(float(eg_data[3]), use_ms=False), seconds_to_td_str(float(eg_data[5]), use_ms=False)]], 'tactiv_ds_eg.tex')

    # t-test for activities durations by module

    activity_times_ttest_ds = calculations['activity_times_ttest_test']
    attt_data = np.array(activity_times_ttest_ds.data).tolist()
    write_file_tex_table(
        [
            [Measurements.name_to_pretty_name(attt_data[0][0]), ">", float(attt_data[0][5]), float(attt_data[0][6])],
            [Measurements.name_to_pretty_name(attt_data[1][0]), ">", float(attt_data[1][5]), float(attt_data[1][6])]
        ],
        "activ_ttest.tex",
        [3],
        lambda v: v<0.05
    )

    # t-test for activities durations by issues (or, some say, by "activity")

    issues_activity_times_ttest_ds = calculations['issues_activity_times_ttest_test']
    iattt_data = np.array(issues_activity_times_ttest_ds.data).tolist()
    write_file_tex_table(
        [
            [iattt_data[0][0].title(), ">", float(iattt_data[0][5]), float(iattt_data[0][6])],
            [iattt_data[1][0].title(), ">", float(iattt_data[1][5]), float(iattt_data[1][6])]
        ],
        "issues_activ_ttest.tex",
        [3],
        lambda v: v<0.05
    )

    # t-test for total activity durations

    total_activity_times_ttest_ds = calculations['total_activity_times_ttest_test']
    tattt_data = np.array(total_activity_times_ttest_ds.data).tolist()
    write_file_tex_table(
        [
            [">", float(tattt_data[0][4]), float(tattt_data[0][5])]
        ],
        "tactiv_ttest.tex",
        [2],
        lambda v: v<0.05
    )

    # tasks descriptive statistics

    task_times_sstats_ds = calculations['task_times_sstats']
    cg_idxs = np.array(task_times_sstats_ds.data)[:,1] == experiment.groups[0].name
    eg_idxs = np.array(task_times_sstats_ds.data)[:,1] == experiment.groups[1].name
    cg_data = np.array(task_times_sstats_ds.data)[cg_idxs].tolist()
    eg_data = np.array(task_times_sstats_ds.data)[eg_idxs].tolist()
    write_file_tex_table(
        [
            [
                "",
                row[0],
                seconds_to_td_str(float(row[2]), use_ms=False),
                seconds_to_td_str(float(row[4]), use_ms=False),
                seconds_to_td_str(float(row[6]), use_ms=False)
            ] for row in cg_data
        ],
        'task_ds_cg.tex')
    write_file_tex_table(
        [
            [
                "",
                row[0],
                seconds_to_td_str(float(row[2]), use_ms=False),
                seconds_to_td_str(float(row[4]), use_ms=False),
                seconds_to_td_str(float(row[6]), use_ms=False)
            ] for row in eg_data
        ],
        'task_ds_eg.tex')

    # t-test for tasks durations

    task_times_ttest_ds = calculations['task_times_ttest_test']
    tttt_data = np.array(task_times_ttest_ds.data).tolist()
    write_file_tex_table(
        [
            [row[0], ">", float(row[5]), float(row[6])]
            for row in tttt_data
        ],
        "task_ttest.tex",
        [3],
        lambda v: v<0.05
    )

    # Descriptive statistics for total task times

    total_task_times_persubject_sstats_ds = calculations['total_task_times_persubject_sstats']
    cg_data = np.array(total_task_times_persubject_sstats_ds.data)[0].tolist()
    eg_data = np.array(total_task_times_persubject_sstats_ds.data)[1].tolist()
    write_file_tex_table([["", seconds_to_td_str(float(cg_data[1]), use_ms=False), seconds_to_td_str(float(cg_data[3]), use_ms=False), seconds_to_td_str(float(cg_data[5]), use_ms=False)]], 'ttaskps_ds_cg.tex')
    write_file_tex_table([["", seconds_to_td_str(float(eg_data[1]), use_ms=False), seconds_to_td_str(float(eg_data[3]), use_ms=False), seconds_to_td_str(float(eg_data[5]), use_ms=False)]], 'ttaskps_ds_eg.tex')

    # t-test for total task durations

    total_task_times_persubject_ttest_ds = calculations['total_task_times_persubject_ttest_test']
    tttttps_data = np.array(total_task_times_persubject_ttest_ds.data).tolist()
    write_file_tex_table(
        [
            [">", float(tttttps_data[0][4]), float(tttttps_data[0][5])]
        ],
        "ttaskps_ttest.tex",
        [2],
        lambda v: v<0.05
    )

    # questionnaire questions lists and statistics tables

    questionnaire_questions = calculations['questionnaire_questions'].data
    write_file_tex_list(questionnaire_questions[:15], "questionnaire_background_questions.tex")
    write_file_tex_list(questionnaire_questions[17:20], "questionnaire_assessment_questions_ef.tex")
    write_file_tex_list(questionnaire_questions[20:24], "questionnaire_assessment_questions_op.tex")
    write_file_tex_list(questionnaire_questions[24:29], "questionnaire_assessment_questions_ia.tex")
    write_file_tex_list(questionnaire_questions[29:33], "questionnaire_assessment_questions_cl.tex")
    write_file_tex_list(questionnaire_questions[33:35], "questionnaire_assessment_questions_un.tex")
    write_file_tex_list(questionnaire_questions[35:37], "questionnaire_assessment_questions_co.tex")

    # questionnaire statistics tables

    questionnaire_histogram_ds = calculations['questionnaire_histogram']
    cg_idxs = np.array(questionnaire_histogram_ds.data)[:,0] == experiment.groups[0].name
    eg_idxs = np.array(questionnaire_histogram_ds.data)[:,0] == experiment.groups[1].name
    cg_data = np.array(questionnaire_histogram_ds.data)[cg_idxs].tolist()
    eg_data = np.array(questionnaire_histogram_ds.data)[eg_idxs].tolist()
    cg_data_dict = dict([(row[1], row[2:]) for row in cg_data])
    eg_data_dict = dict([(row[1], row[2:]) for row in eg_data])
    max_value = 14 # TODO: calculate the highest histogram value rather than have it hardcoded

    questionnaire_sstats_ds = calculations['questionnaire_sstats']
    cg_idxs = np.array(questionnaire_sstats_ds.data)[:,1] == experiment.groups[0].name
    eg_idxs = np.array(questionnaire_sstats_ds.data)[:,1] == experiment.groups[1].name
    cg_data = np.array(questionnaire_sstats_ds.data)[cg_idxs].tolist()
    eg_data = np.array(questionnaire_sstats_ds.data)[eg_idxs].tolist()
    cg_sstats_dict = dict([(row[0], row[2:]) for row in cg_data])
    eg_sstats_dict = dict([(row[0], row[2:]) for row in eg_data])
    question_numbers = np.array(cg_data).transpose()[0]

    questionnaire_hypothesis_ds = calculations['questionnaire_hypothesis']
    questionnaire_hypothesis_dict = dict(questionnaire_hypothesis_ds.data)

    questionnaire_mww_test_ds = calculations['questionnaire_mww_test']
    questionnaire_mww_test = dict([(a[0], a[1:]) for a in questionnaire_mww_test_ds.data])

    questionnaire_table = list()
    for question_number in question_numbers:
        questionnaire_table.append([
            question_number,
            """\hspace{{0cm}}\\ratiofivespark{{{0}}}{{{1}}}{{{2}}}{{{3}}}{{{4}}}{{{5}}}""".format(*tuple([max_value] + cg_data_dict[question_number])),
            float(cg_sstats_dict[question_number][2]),
            float(cg_sstats_dict[question_number][4]),
            """\hspace{{0cm}}\\ratiofivespark{{{0}}}{{{1}}}{{{2}}}{{{3}}}{{{4}}}{{{5}}}""".format(*tuple([max_value] + eg_data_dict[question_number])),
            float(eg_sstats_dict[question_number][2]),
            float(eg_sstats_dict[question_number][4]),
            {u'\u2260': "$\\neq$", "<": "<", ">": ">"}.get(questionnaire_hypothesis_dict[question_number], "\\hl{?}"),
            float(questionnaire_mww_test[question_number][0]),
            float(questionnaire_mww_test[question_number][ {u'\u2260': 1, "<": 2, ">": 3}.get(questionnaire_hypothesis_dict[question_number], None)]),
        ])
    write_file_tex_table(questionnaire_table[:15], "questionnaire_background.tex", [9], lambda v: v<0.05)
    write_file_tex_table(questionnaire_table[15:18], "questionnaire_assessment_ef.tex", [9], lambda v: v<0.05)
    write_file_tex_table(questionnaire_table[18:22], "questionnaire_assessment_op.tex", [9], lambda v: v<0.05)
    write_file_tex_table(questionnaire_table[22:27], "questionnaire_assessment_ia.tex", [9], lambda v: v<0.05)
    write_file_tex_table(questionnaire_table[27:31], "questionnaire_assessment_cl.tex", [9], lambda v: v<0.05)
    write_file_tex_table(questionnaire_table[31:33], "questionnaire_assessment_un.tex", [9], lambda v: v<0.05)
    write_file_tex_table(questionnaire_table[33:35], "questionnaire_assessment_co.tex", [9], lambda v: v<0.05)

    # questionnaire full answers
    questionnaire_one_answer_per_column = calculations['questionnaire_one_answer_per_column'].data
    write_file_tex_table(questionnaire_one_answer_per_column, "questionnaire_assessment_all_answers.tex")

    # questionnaire levene tests
    questionnaire_levene = calculations['questionnaire_levene_test'].data
    write_file_tex_table([(q,float(w),float(p)) for (q,p,w) in questionnaire_levene], "questionnaire_levene.tex")

    # questionnaire histograms
    def get_apx_tex_histogram_hists(q_nbr, q_txt, counter):
        return r"""
    \begin{figure}[!h]
            \centering
            \begin{subfigure}[b]{0.5\textwidth}
              \centering
              \includegraphics[scale=0.3]{asaanalyser/asaanalyser/out/experiments/experiment03/pdf/{questionnaires_""" + str(counter) + """_""" + q_nbr + r"""_e3_control_group}.pdf}
              \caption{Control Group}
              \label{fig:histogram-""" + q_nbr + r"""-cg}
            \end{subfigure}%
            ~ %add desired spacing between images, e. g. ~, \quad, \qquad etc. (or a blank line to force the subfigure onto a new line)
            \begin{subfigure}[b]{0.5\textwidth}
              \centering
              \includegraphics[scale=0.3]{asaanalyser/asaanalyser/out/experiments/experiment03/pdf/{questionnaires_""" + str(counter+1) + """_""" + q_nbr + """_e3_experimental_group}.pdf}
              \caption{Experimental Group}
              \label{fig:histogram-""" + q_nbr + """-eg}
            \end{subfigure}
            \caption[Histogram of the answers to the questionnaire item """ + q_nbr + """]{Histogram of the answers to the questionnaire item """ + q_nbr + """ --- \emph{""" + q_txt + """}}
            \label{fig:histogram-""" + q_nbr + """}
    \end{figure}

    """

    def get_apx_tex_hist_label(q_nbr, grp_two_letter_code):
        return r"""fig:histogram-""" + q_nbr + """-""" + grp_two_letter_code

    tex_file_questionnaire_hist = open(os.path.join(output_path_tex, "questionnaire_histograms.tex"), "w")
    tex_file_questionnaire_hist_labels = open(os.path.join(output_path_tex, "questionnaire_histograms_labels_refs.tex"), "w")
    questionnaire_histogram_qnbrs = asa_calcs.get_calculation("questionnaire_histogram")[:,1].tolist()
    counter = 1
    for q_nbr, q_txt in questionnaire_questions:
        if not q_nbr in questionnaire_histogram_qnbrs: continue
        tex_file_questionnaire_hist.write(get_apx_tex_histogram_hists(q_nbr, q_txt, counter))
        if (counter+1-4) % 6 == 0:
            tex_file_questionnaire_hist.write(r"\clearpage" + "\n")
        if counter > 1: tex_file_questionnaire_hist_labels.write(",")
        tex_file_questionnaire_hist_labels.write(
            get_apx_tex_hist_label(q_nbr, "cg") + "," +
            get_apx_tex_hist_label(q_nbr, "eg")
        )
        counter += 2
    tex_file_questionnaire_hist.close()

    ############

    def round_seconds_to_the_next_even_minute(seconds):
        if math.isnan(seconds): return seconds
        td = timedelta(seconds=seconds)
        mins = int(td.seconds/60)+1
        if mins%2!=0:
            mins+=1
        return timedelta(days=td.days, minutes=mins).total_seconds()

    #### Draw box-and-whiskers for times. One boxplot per group and task/activity-type
    def plot_boxplot(calc_data, time_grouper, output_filename_pdf):
        fig = plt.figure()
        from pandas import DataFrame
        calc_data_df = DataFrame(calc_data)
        groups_values = []
        xlabels_box = []
        xlabels_boxgroup = []

        grouping_func = time_grouper.get_grouping_func()
        for xlabel, group, times in grouping_func(calc_data_df):
            groups_values.append([time.total_seconds() for time in times])
            if group.endswith("control_group"):
                group = "CG"
            elif group.endswith("experimental_group"):
                group = "EG"
            xlabels_box.append(group)
            if not xlabel is None and not xlabel in xlabels_boxgroup:
                xlabels_boxgroup.append(xlabel)

        ax = fig.add_subplot(111)
        ax.boxplot(groups_values) #, whis=3

        # ticks and labels
        ax.set_xticks(np.arange(len(xlabels_box)) + 1)
        ax.set_xticklabels(xlabels_box) #, rotation=45, ha='right')

        boxesgroup_ticks = [1.5+i*2 for i in range((len(xlabels_box)/2))]
        ax2 = ax.twiny()
        ax2.xaxis.tick_bottom()
        ax2.xaxis.set_label_position('bottom')
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(boxesgroup_ticks)
        ax2.set_xticklabels(xlabels_boxgroup)
        for mtickline in ax2.xaxis.get_ticklines():
            mtickline.set_visible(False)

        # separating lines
        for i in range((len(xlabels_box)/2)-1):
            ax.axvline(x=2.5+i*2, visible=True, color='lightgrey')
        #ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

        lengths = [len(row) for row in groups_values]
        indexes = range(len(groups_values))
        xs = [[t[1] + 1] * t[0] for t in zip(lengths, indexes)]
        xs = [x for row in xs for x in row]
        ys = [y for row in groups_values for y in row]
        ax.plot(xs, ys, '.g') # draw data points

        yticks = np.linspace(0, round_seconds_to_the_next_even_minute(max([v for vrow in groups_values for v in vrow]) * 1.01), 7)
        ylabels = tuple(td_to_str(timedelta(seconds=t), False) for t in yticks)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        #ax.set_aspect(0.2/100)

        ax.tick_params(axis='x', pad=10)
        ax.tick_params(axis='y', pad=10)
        ax2.tick_params(axis='x', pad=35)

        fig.set_tight_layout(True)
        fig.subplots_adjust(bottom=0.15)

        plt.savefig(output_filename_pdf)

    class BoxplotDataTimeGrouper(object): pass
    class BoxplotDataTimeGrouperByGroupsAndTasks(BoxplotDataTimeGrouper):
        # generator function that groups times by experimental group and task
        def _timesforgroupsandtasks(self, times_dataframe):
            for (task, group), times in [(n, g.values[:, 3:].transpose()[0].tolist()) for n, g in times_dataframe.groupby([2, 0], as_index=False)]:
                    yield (task.upper(), group, times)
        def get_grouping_func(self):
            return self._timesforgroupsandtasks

    class BoxplotDataTimeGrouperByGroups(BoxplotDataTimeGrouper):
        def __init__(self, data_columns):
            self.data_columns = data_columns

        # generator function that groups times by experimental group
        def _timesforgroups(self, times_dataframe):
            for group, times in [(n, g.values[:, self.data_columns[0]:self.data_columns[1]].transpose()[0].tolist()) for n, g in times_dataframe.groupby(0, as_index=False)]:
                    yield (None, group, times)
        def get_grouping_func(self):
            return self._timesforgroups

    class BoxplotDataTimeGrouperByGroupsAndActivities(BoxplotDataTimeGrouper):
        def __init__(self, activity_idxs):
            self.activity_idxs = activity_idxs # list of tuples (idx, name)

        # generator function that groups times by group and kind of activity
        def _timesforgroupsandactivities(self, times_dataframe):
            for activity_idx, activity_name in self.activity_idxs:
                for group, times in [(n,g.values[:,activity_idx+2:activity_idx+3].transpose()[0].tolist()) for n,g in times_dataframe.groupby([0], as_index=False)]:
                    yield (Measurements.name_to_pretty_name(activity_name), group, times)

        def get_grouping_func(self):
            return self._timesforgroupsandactivities

    def plot_tsktimes_by_task_boxplot(asa_calcs, output_path_pdf):
        calc_data = asa_calcs.get_calculation('task_times', lambda x: (x[2], x[0]))
        plot_boxplot(calc_data, BoxplotDataTimeGrouperByGroupsAndTasks(), os.path.join(output_path_pdf, "boxplot_task_times_by_task.pdf"))

    def plot_acttimes_by_activity_boxplot(asa_calcs, activity_idxs, output_filename_pdf):
        calc_data = asa_calcs.get_calculation('activity_times', lambda x: x[0])
        plot_boxplot(calc_data, BoxplotDataTimeGrouperByGroupsAndActivities(activity_idxs), output_filename_pdf)

    plot_tsktimes_by_task_boxplot(asa_calcs, output_path_pdf)

    activity_idxs = Measurements.to_ids_list_that_matter() # excluding uninsteresting activities (mostly 'edit' activities)
    plot_acttimes_by_activity_boxplot(asa_calcs, activity_idxs, os.path.join(output_path_pdf, "boxplot_activity_times_by_activity.pdf"))
    for act_idx_tpl in activity_idxs:
        plot_acttimes_by_activity_boxplot(asa_calcs, [act_idx_tpl], os.path.join(output_path_pdf, "boxplot_activity_times_by_activity_{0}.pdf".format(act_idx_tpl[0])))

    calc_data = asa_calcs.get_calculation('task_times', lambda x: (x[2], x[0]))
    plot_boxplot(calc_data, BoxplotDataTimeGrouperByGroups((3,4)), os.path.join(output_path_pdf, "boxplot_total_task_times.pdf"))

    calc_data = asa_calcs.get_calculation('total_task_times_persubject', lambda x: x[0])
    plot_boxplot(calc_data, BoxplotDataTimeGrouperByGroups((2,3)), os.path.join(output_path_pdf, "boxplot_total_task_times_persubject.pdf"))

    calc_data = asa_calcs.get_calculation('total_activity_times', lambda x: x[0])
    plot_boxplot(calc_data, BoxplotDataTimeGrouperByGroups((2,3)), os.path.join(output_path_pdf, "boxplot_total_activity_times_persubject.pdf"))


    #### Draw bar chart of the average times for each task, for each group

    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.direction'] = 'out'


    def plot_total_task_times_bars(asa_calcs, sstat_type, output_filename_pdf):
        """
        :param sstat_type: refers to column index of the "simple stat" to plot.
            1 - sum
            3 - mean
            4 - median
            ...
        """
        total_task_times_sstats = asa_calcs.get_calculation("total_task_times_sstats", lambda x: x[0])
        groups_data = asa_calcs.experiment.get_groups(total_task_times_sstats, columns=(sstat_type, sstat_type+1), group_column=0)
        group1 = [float(x) for x in groups_data[0][0].tolist()]
        group2 = [float(x) for x in groups_data[1][0].tolist()]
        group1_name = asa_calcs.experiment.groups[0].name
        group2_name = asa_calcs.experiment.groups[1].name
        plot_durations_bars(group1_name, group1, group2_name, group2, [""], output_filename_pdf)


    def plot_task_times_sstats_bars(asa_calcs, sstat_type, output_filename_pdf):
        """
        :param sstat_type: refers to column index of the "simple stat" to plot.
            4 - mean
            5 - median
            ...
        """
        task_times_sstats = asa_calcs.get_calculation("task_times_sstats", lambda x: x[0])
        groups_data = asa_calcs.experiment.get_groups(task_times_sstats, columns=(sstat_type, sstat_type+1), group_column=1)
        group1 = [float(x) for x in groups_data[0][0].tolist()]
        group2 = [float(x) for x in groups_data[1][0].tolist()]
        group1_name = asa_calcs.experiment.groups[0].name
        group2_name = asa_calcs.experiment.groups[1].name
        xaxis_labels = np.unique(task_times_sstats[:, 0].transpose())
        plot_durations_bars(group1_name, group1, group2_name, group2, xaxis_labels, output_filename_pdf)

    def plot_activity_times_sstats_bars(asa_calcs, sstat_type, output_filename_pdf):
        """
        :param sstat_type:
         1 - sum
         3 - mean
         4 - median
         5 - stddev
         6 - var
        """
        activity_times_sstats = asa_calcs.get_calculation("activity_times_sstats") # assuming everything is ordered right
        groups_data = asa_calcs.experiment.get_groups(activity_times_sstats, columns=(sstat_type, sstat_type+1), group_column=0)
        group1 = [float(x) for x in groups_data[0][0].tolist()]
        group2 = [float(x) for x in groups_data[1][0].tolist()]
        group1_name = asa_calcs.experiment.groups[0].name
        group2_name = asa_calcs.experiment.groups[1].name
        def get_unique_with_order(items):
            seen = set()
            unique_items = list()
            for item in items:
                if not item in seen: unique_items.append(item)
                seen.add(item)
            return unique_items
        xaxis_labels = [Measurements.name_to_pretty_name(xal) for xal in get_unique_with_order(activity_times_sstats[:, 1:2].transpose()[0])]
        plot_durations_bars(group1_name, group1, group2_name, group2, xaxis_labels, output_filename_pdf, False)

    def plot_durations_bars(group1_name, group1_values, group2_name, group2_values, xaxis_labels, output_filename_pdf, rotate_xaxis_labels=False):
        bar_x_pos = np.arange(max(len(group1_values), len(group2_values)))
        bar_x_pos = bar_x_pos / 1.2
        bar_width = 0.25

        fig = plt.figure(edgecolor='white')
        ax = fig.add_subplot(111)
        rects1 = ax.bar(bar_x_pos + bar_width, group1_values, bar_width, color='#D94527', edgecolor='white')
        rects2 = ax.bar(bar_x_pos + bar_width * 2, group2_values, bar_width, color='#D99E27', edgecolor='white')

        ax.set_ylabel('Duration (h:m:s)')
        ax.yaxis.labelpad = 20
        yticks = np.linspace(0, round_seconds_to_the_next_even_minute(max(group1_values + group2_values) * 1.15), 7)
        ylabels = tuple(td_to_str(timedelta(seconds=t), False) if not math.isnan(t) else None for t in yticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', pad=10)
        ax.set_yticklabels(ylabels)


        ax.set_xticks(bar_x_pos + bar_width * 2)
        ax.tick_params(axis='x', pad=10)
        if rotate_xaxis_labels:
            ax.set_xticklabels(xaxis_labels, rotation=45, ha='right')
            fig.subplots_adjust(bottom=0.3)
        else:
            ax.set_xticklabels(xaxis_labels)
        ax.set_axisbelow(True)
        fig.gca().yaxis.grid(True)

        assert(group1_name.endswith("control_group"))
        assert(group2_name.endswith("experimental_group"))

        plt.subplots_adjust(left=0.17)
        leg = ax.legend(("Control Group", "Experimental Group"), loc='upper right', prop={'size': 12}, )
        #leg.get_frame().set_facecolor()
        leg.get_frame().set_edgecolor('white')

        ax = fig.gca()
        ax.spines["top"].set_linewidth(0)
        ax.spines["right"].set_linewidth(0)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig(output_filename_pdf)

    plot_total_task_times_bars(asa_calcs, 1, os.path.join(output_path_pdf, "bars_total_task_times.pdf"))
    plot_task_times_sstats_bars(asa_calcs, 4, os.path.join(output_path_pdf, "bars_task_times_by_task_sstats_mean.pdf"))
    plot_task_times_sstats_bars(asa_calcs, 5, os.path.join(output_path_pdf, "bars_task_times_by_task_sstats_median.pdf"))
    plot_task_times_sstats_bars(asa_calcs, 6, os.path.join(output_path_pdf, "bars_task_times_by_task_sstats_std.pdf"))
    plot_task_times_sstats_bars(asa_calcs, 7, os.path.join(output_path_pdf, "bars_task_times_by_task_sstats_var.pdf")) # TODO: Figure out how to handle units of the Y axis; they should be the duration square. See http://en.wikipedia.org/wiki/Variance#Units_of_measurement


    # Média dos tempos que os pares demoraram em cada actividade
    plot_activity_times_sstats_bars(asa_calcs, 2, os.path.join(output_path_pdf, "bars_activity_times_by_activity_sstats_sum.pdf"))
    plot_activity_times_sstats_bars(asa_calcs, 4, os.path.join(output_path_pdf, "bars_activity_times_by_activity_sstats_mean.pdf"))
    plot_activity_times_sstats_bars(asa_calcs, 5, os.path.join(output_path_pdf, "bars_activity_times_by_activity_sstats_median.pdf"))


    # Plotting questionnaire data as a bars chart
    def plot_questionnaire_sstats_bars(asa_calcs, sstat_type, output_filename_pdf):
        """
        :param sstat_type: refers to column index of the "simple stat" to plot.
            4 - mean
            5 - median
            ...
        """
        questionnaire_sstats = asa_calcs.get_calculation("questionnaire_sstats", lambda x: x[0])
        if len(questionnaire_sstats) == 0:
            print "Warning: skipped plotting questionnaire sstats bars for sstat #{0} of experiment '{1}'. No data provided.".format(sstat_type, asa_calcs.experiment.name)
            return
        groups_data = asa_calcs.experiment.get_groups(questionnaire_sstats, columns=(sstat_type, sstat_type+1), group_column=1)
        group1 = [float(x) for x in groups_data[0][0].tolist()]
        group2 = [float(x) for x in groups_data[1][0].tolist()]
        group1_name = asa_calcs.experiment.groups[0].name
        group2_name = asa_calcs.experiment.groups[1].name
        xaxis_labels = np.unique(questionnaire_sstats[:, 0].transpose())
        plot_questionnaire_sstats_bars2(group1_name, group1, group2_name, group2, xaxis_labels, output_filename_pdf)

    def plot_questionnaire_sstats_bars2(group1_name, group1_values, group2_name, group2_values, xaxis_labels, output_filename_pdf, rotate_xaxis_labels=False):
        bar_x_pos = np.arange(max(len(group1_values), len(group2_values)))
        bar_x_pos = bar_x_pos / 1.2
        bar_width = 0.25

        fig = plt.figure(edgecolor='white')
        ax = fig.add_subplot(111)
        rects1 = ax.bar(bar_x_pos + bar_width, group1_values, bar_width, color='#D94527', edgecolor='white')
        rects2 = ax.bar(bar_x_pos + bar_width * 2, group2_values, bar_width, color='#D99E27', edgecolor='white')

        ax.set_ylabel('Answer (1-5)')
        ax.yaxis.labelpad = 20
        yticks = np.linspace(0, 5, 6)
        ylabels = tuple(yticks)
        ax.set_yticks(yticks)
        ax.tick_params(axis='y', pad=10)
        ax.set_yticklabels(ylabels)

        ax.set_xticks(bar_x_pos + bar_width * 2)
        ax.tick_params(axis='x', pad=10)
        if rotate_xaxis_labels:
            ax.set_xticklabels(xaxis_labels, rotation=45, ha='right')
            fig.subplots_adjust(bottom=0.3)
        else:
            ax.set_xticklabels(xaxis_labels)
        ax.set_axisbelow(True)
        fig.gca().yaxis.grid(True)

        plt.subplots_adjust(left=0.17)
        leg = ax.legend((group1_name, group2_name), loc='upper right', prop={'size': 12}, )
        #leg.get_frame().set_facecolor()
        leg.get_frame().set_edgecolor('white')

        ax = fig.gca()
        ax.spines["top"].set_linewidth(0)
        ax.spines["right"].set_linewidth(0)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.savefig(output_filename_pdf)

    plot_questionnaire_sstats_bars(asa_calcs, 4, os.path.join(output_path_pdf, "questionnaires_sstats_mean.pdf"))
    plot_questionnaire_sstats_bars(asa_calcs, 5, os.path.join(output_path_pdf, "questionnaires_sstats_median.pdf"))


    # Plotting questionnaire data as a bars chart
    def plot_questionnaire_answers_bars(asa_calcs, output_filename_pdf):
        questionnaire_histogram = asa_calcs.get_calculation("questionnaire_histogram")
        if len(questionnaire_histogram) == 0:
            print "Warning: skipped plotting questionnaire histogram bars of experiment '{0}'. No data provided.".format(asa_calcs.experiment.name)
            return

        def to_intlist(full_np_row):
            return full_np_row[2:].astype(np.int).tolist()

        c = 1
        for prev_row,row,next_row in neighborhood(questionnaire_histogram):
            if not prev_row is None and row[1] == prev_row[1]:
                max_count = max(to_intlist(row) + to_intlist(prev_row))
            elif not next_row is None and row[1] == next_row[1]:
                max_count = max(to_intlist(row) + to_intlist(next_row))
            else:
                max_count = max(to_intlist(row))
                print "Warning: Couldn't find the other group for question {0} (found only {1}).".format(row[1], row[0])

            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.bar(np.arange(1,6), row[2:].astype(np.int), color='0.50', align='center')

            ax.set_xlim(0.5, 5.5) # 1-5 likert scale
            ax.set_ylim(0, max_count+0.1) # Adding 0.1 allows space for a thicker top grid line
            ax.yaxis.set_major_locator(MaxNLocator(prune='lower', integer=True)) # remove 0 from the y axis
            ax.yaxis.grid(True, which='major', linewidth=4)
            ax.spines['right'].set_color('none')
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.tick_params(axis='y', pad=26)
            ax.tick_params(axis='x', pad=10)

            plt.tick_params(axis='x', which='major', labelsize=34)
            plt.tick_params(axis='y', which='major', labelsize=28)
            for mtickline in ax.xaxis.get_ticklines():
                mtickline.set_visible(False)
            for mtickline in ax.yaxis.get_ticklines():
                mtickline.set_visible(False)

            plt.subplots_adjust(bottom=0.13)
            plt.savefig("{0}_{1}_{2}_{3}.{4}".format(output_filename_pdf[:-4], c, row[1], row[0], output_filename_pdf[-3:]))
            c += 1

    plot_questionnaire_answers_bars(asa_calcs, os.path.join(output_path_pdf, "questionnaires.pdf"))

    stdout.write(" [finished]")
    stdout.flush()
