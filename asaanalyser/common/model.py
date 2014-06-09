# -*- coding: utf-8 -*-
from sys import stdout
import sqlite3
import math
import csv
from datetime import timedelta
from util import parse_date, total_seconds, groupby

CONSIDER_ONLY_COMPLETED_TASKS = False # this is ok because nothing significant is missing from incomplete tasks

class ASADatabase(object):
    def __init__(self, db_filename):
        """db_filename: the database filename to open with full or relative path"""
        self.db_filename = db_filename

    def _get_full_iterable_data(self):
        conn = sqlite3.connect(self.db_filename)
        c = conn.cursor()
        stats_table_present = False
        c.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='asa_accurate_analytics';")
        if c.fetchone()[0] == 1:
            stats_table_present = True
        c.close()

        if stats_table_present:
            c = conn.cursor()
            c.execute('SELECT id, resource_type, resource_id, operation, username, time_started, time_ended '
                      'FROM asa_accurate_analytics '
                      'WHERE NOT time_ended IS NULL')
            result = c.fetchall()
            c.close()
            conn.close()
        else:
            result = []
        return result

    def get_full_dataset(self):
        return ASADBDataSet(self._get_full_iterable_data())


class FakeASADatabase(object):
    def __init__(self, csv_filename):
          self.csv_filename = csv_filename
    def get_full_dataset(self):
        data = []
        with open(self.csv_filename, 'rbU') as csvfile:
            reader = csv.reader(csvfile)
            reader.next() # skip the headers row
            data = list(reader)
        return ASADBDataSet(data)

class ResourceTypes:
    wiki = "wiki"
    search = "search"
    asa_artifact = "asa_artifact"
    asa_spec = "asa_spec"
    asa_index = "index"
    asa_search = "asa_search"

class Measurements:
    wiki_view = 'wiki_view'
    wiki_edit = 'wiki_edit'
    search = 'search'
    asa_artifact_view = 'asa_artifact_view'
    asa_artifact_edit = 'asa_artifact_edit'
    asa_index = 'index_view'
    asa_search = 'asa_search_view'

    pretty_names = {wiki_view: "Wiki Pages", search: "Search", asa_artifact_view: "ASA Artifacts", asa_index: "ASA Index"}

    @classmethod
    def to_list(cls):
        return [cls.wiki_view, cls.wiki_edit, cls.search, cls.asa_artifact_view, cls.asa_artifact_edit, cls.asa_index, cls.asa_search]

    @classmethod
    def to_ids_list_that_matter(cls):
        return [(0, cls.wiki_view), (2, cls.search), (3, cls.asa_artifact_view), (5, cls.asa_index)]

    @classmethod
    def name_to_pretty_name(cls, name):
        return cls.pretty_names[name]


class ASAExperiment(object):
    def __init__(self, name, groups, questionnaire_hypos, questionnaire_questions):
        self.name = name
        self.groups = groups
        assert(len(groups)==2) # For simplicity sake. May eventually be made more generic.
        self.questionnaire_hypos = questionnaire_hypos
        self.questionnaire_questions = questionnaire_questions

    def _get_groups_in_seconds(self, data_array, columns=(3,None)):
        """Gets dates in seconds for each of the experiment's groups"""
        groups = []
        for group in self.get_groups(data_array, columns):
            group = [[total_seconds(time) for time in dimension] for dimension in group]
            groups.append(group)
        return groups

    def get_groups(self, data_array, columns=(3,None), group_column=0):
        """Partitions the data in the specified columns by the specified groups"""
        groups_names = [g.name for g in self.groups]
        groups = []
        for group_name in groups_names:
            group = data_array[:, group_column] == group_name
            group = data_array[group][:, columns[0]:columns[1]].transpose()
            groups.append(group)
        return groups

    def get_questionnaire_hypothesis(self):
        if self.questionnaire_hypos is None: return None
        if len(self.questionnaire_hypos)==0: return []
        return self.questionnaire_hypos[1]

    def get_questionnaire_questions(self):
        if self.questionnaire_questions is None: return None
        if len(self.questionnaire_questions)==0: return []
        return self.questionnaire_questions[1]

    def run_tests(self):
        """Processes all the data, runs the statistical tests, and returns the results"""
        import numpy as np
        from stats import get_mww, get_ttest_equal_var, get_ttest_diff_var, get_levene, get_simple_stats, get_shapiro

        calculations = []

        headers_task_durations = ['group', 'user', 'task', 'duration']
        headers_activity_times = ['group', 'user'] + Measurements.to_list()
        tasks_data = ASADataSet(headers_task_durations, [])
        tasks_data_secs = ASADataSet(headers_task_durations, [])
        activities_data = ASADataSet(headers_activity_times, [])
        activities_data_secs = ASADataSet(headers_activity_times, [])
        stdout.write("Running tests for experiment '{0}'\n".format(self.name))
        stdout.write("Loading tasks: ")
        stdout.flush()
        for group in self.groups:
            tasks_rows = [(group.name,) + row for row in group.get_task_times()]
            tasks_data.data.extend(tasks_rows)
            tasks_data_secs.data.extend([row[0:3] + tuple(total_seconds(v) for v in row[3:]) for row in tasks_rows])
            stdout.write(".")
            stdout.flush()
        calculations.append(("task_times", tasks_data))
        calculations.append(("task_times_secs", tasks_data_secs))
        stdout.write(" [finished]\n")
        stdout.flush()

        stdout.write("Loading activities: ")
        stdout.flush()
        for group in self.groups:
            activities_rows = [[group.name] + row for row in group.get_activity_times()]
            activities_data.data.extend(activities_rows)
            activities_data_secs.data.extend([row[0:2] + [total_seconds(v) for v in row[2:]] for row in activities_rows])
            stdout.write(".")
            stdout.flush()
        calculations.append(("activity_times", activities_data))
        calculations.append(("activity_times_secs", activities_data_secs))
        stdout.write(" [finished]\n")
        stdout.flush()


        ###### Run statistical tests ######
        stdout.write("Running statistical tests")
        stdout.flush()

        tasks_times_array = np.array(tasks_data.data)
        activity_times_array = np.array(activities_data.data)

        sstats_headers = ['group', 'sum', 'average', 'mean', 'median', 'std', 'var', 'count']
        mww_headers = ['u', 'p_value_twotailed', 'p_value_lessthan', 'p_value_greaterthan']
        ttest_headers = ['t_twotailed', 'p_value_twotailed', 't_lessthan', 'p_value_lessthan', 't_greaterthan', 'p_value_greaterthan', 'for_variance']
        levene_headers = ['p_value', 'w']
        shapiro_headers = ['group', 'activity', 'p_value', 'w']

        ### task tests

        sstats_results = ASADataSet(['task'] + sstats_headers, [])
        mww_results = ASADataSet(['task'] + mww_headers, [])
        ttest_results = ASADataSet(['task'] + ttest_headers, [])
        levene_results = ASADataSet(['task'] + levene_headers, [])

        tasks_names = sorted(set(tasks_times_array[:, 2]))
        for task_name in tasks_names:
            task_array = tasks_times_array[tasks_times_array[:, 2] == task_name]
            groups_data = self._get_groups_in_seconds(task_array)
            group1_data = groups_data[0][0] # one "dimension" assumed, as task times imply only one column
            group2_data = groups_data[1][0] # one "dimension" assumed, as task times imply only one column

            sstats_results.data.append((task_name, self.groups[0].name) + get_simple_stats(group1_data))
            sstats_results.data.append((task_name, self.groups[1].name) + get_simple_stats(group2_data))
            mww_results.data.append((task_name,) + get_mww(group1_data, group2_data))

            tasks_levene_result = get_levene(group1_data, group2_data)
            levene_results.data.append((task_name,) + tasks_levene_result)
            if tasks_levene_result[0] > 0.05: # equal variance
                ttest_results.data.append((task_name,) + get_ttest_equal_var(group1_data, group2_data) + ("equal",))
            else:
                ttest_results.data.append((task_name,) + get_ttest_diff_var(group1_data, group2_data) + ("diff",))


        calculations.append(("task_times_sstats", sstats_results))
        calculations.append(("task_times_mww_test", mww_results))
        calculations.append(("task_times_ttest_test", ttest_results))
        calculations.append(("task_times_levene_test", levene_results))

        #### totals task times tests

        groups_data = self._get_groups_in_seconds(tasks_times_array)
        group1_data = groups_data[0][0] # one "dimension" assumed, as task times imply only one column
        group2_data = groups_data[1][0] # one "dimension" assumed, as task times imply only one column
        calculations.append(("total_task_times_sstats", ASADataSet(sstats_headers, [(self.groups[0].name,) + get_simple_stats(group1_data), (self.groups[1].name,) + get_simple_stats(group2_data)])))
        calculations.append(("total_task_times_mww_test", ASADataSet(mww_headers, [get_mww(group1_data, group2_data)])))
        total_levene_result = [get_levene(group1_data, group2_data)]
        calculations.append(("total_task_times_levene_test", ASADataSet(levene_headers, total_levene_result)))
        if total_levene_result[0] > 0.05: # equal variance
            calculations.append(("total_task_times_ttest_test", ASADataSet(ttest_headers, [get_ttest_equal_var(group1_data, group2_data) + ("equal",)])))
        else:
            calculations.append(("total_task_times_ttest_test", ASADataSet(ttest_headers, [get_ttest_diff_var(group1_data, group2_data) + ("diff",)])))


        #### totals task times tests per subject
        ### (i.e., times that subjects took working on the entirety of the tasks, rather than the times they took on each task)

        from pandas import DataFrame
        total_task_times = np.array(DataFrame([row[0:2] + row[3:] for row in tasks_times_array.tolist()]).groupby([0,1], as_index=False).aggregate(np.sum).to_records(index=False).tolist())
        calculations.append(("total_task_times_persubject", ASADataSet(['group', 'user', 'duration'], total_task_times)))

        groups_data = self._get_groups_in_seconds(total_task_times, columns=(2,3))
        group1_data = groups_data[0][0] # one "dimension" assumed, as task times imply only one column
        group2_data = groups_data[1][0] # one "dimension" assumed, as task times imply only one column
        calculations.append(("total_task_times_persubject_sstats", ASADataSet(sstats_headers,
            [(self.groups[0].name,) + get_simple_stats(group1_data), (self.groups[1].name,) + get_simple_stats(group2_data)])))
        calculations.append(("total_task_times_persubject_mww_test", ASADataSet(mww_headers, [get_mww(group1_data, group2_data)])))
        total_levene_result = [get_levene(group1_data, group2_data)]
        calculations.append(("total_task_times_persubject_levene_test", ASADataSet(levene_headers, total_levene_result)))
        if total_levene_result[0] > 0.05: # equal variance
            calculations.append(("total_task_times_persubject_ttest_test", ASADataSet(ttest_headers, [get_ttest_equal_var(group1_data, group2_data) + ("equal",)])))
        else:
            calculations.append(("total_task_times_persubject_ttest_test", ASADataSet(ttest_headers, [get_ttest_diff_var(group1_data, group2_data) + ("diff",)])))


        #### activity tests

        # [group, user, wiki_view, wiki_edit, search, asa_artifact_view, asa_artifact_edit, asa_index, asa_search]
        groups_data = self._get_groups_in_seconds(activity_times_array, columns=(2,None))
        group1_data = groups_data[0]
        group2_data = groups_data[1]
        intermediate_calcs = {
            "activity_times_sstats": ASADataSet(sstats_headers[:1] + ['activity'] + sstats_headers[1:], []),
            "activity_times_shapiro_test": ASADataSet(shapiro_headers, []),
            "activity_times_mww_test": ASADataSet(['activity'] + mww_headers, []),
            "activity_times_ttest_test": ASADataSet(['activity'] + ttest_headers, []),
            "activity_times_levene_test": ASADataSet(['activity'] + levene_headers, []),
        }
        for measurement_id, measurement in Measurements.to_ids_list_that_matter():
            intermediate_calcs["activity_times_sstats"].data.extend(
                [
                    (self.groups[0].name, measurement) + get_simple_stats(group1_data[measurement_id]),
                    (self.groups[1].name, measurement) + get_simple_stats(group2_data[measurement_id])
                ]
            )

            import warnings
            with warnings.catch_warnings(record=True) as w: # catch warnings
                intermediate_calcs["activity_times_shapiro_test"].data.append(
                    (self.groups[0].name, measurement) + get_shapiro(group1_data[measurement_id])
                )
                if len(w) > 0:
                    print '\x1b[31m' + "\n... Warning running shapiro-wilk on '{0}' for group '{1}': {2}".format(measurement, self.groups[0].name, w[-1].message) + '\033[0m'
            with warnings.catch_warnings(record=True) as w: # catch warnings
                intermediate_calcs["activity_times_shapiro_test"].data.append(
                    (self.groups[1].name, measurement) + get_shapiro(group2_data[measurement_id])
                )
                if len(w) > 0:
                    print '\x1b[31m' + "\n... Warning running shapiro-wilk on '{0}' for group '{1}': {2}".format(measurement, self.groups[1].name, w[-1].message) + '\033[0m'

            try:
                intermediate_calcs["activity_times_mww_test"].data.append(
                    (measurement,) + get_mww(group1_data[measurement_id], group2_data[measurement_id])
                )
            except ValueError:
                # get_mww() returns a ValueError when the values on both groups are the same
                print "MWW raised a ValueError. Values on both groups are the same?"
                intermediate_calcs["activity_times_mww_test"].data.append((measurement, None, None))

            activities_levene_result = get_levene(group1_data[measurement_id], group2_data[measurement_id])
            intermediate_calcs["activity_times_levene_test"].data.append(
                (measurement,) + activities_levene_result
            )
            if activities_levene_result[0] > 0.05: # equal variance
                intermediate_calcs["activity_times_ttest_test"].data.append(
                    (measurement,) + get_ttest_equal_var(group1_data[measurement_id], group2_data[measurement_id]) + ("equal",)
                )
            else:
                intermediate_calcs["activity_times_ttest_test"].data.append(
                    (measurement,) + get_ttest_diff_var(group1_data[measurement_id], group2_data[measurement_id]) + ("diff",)
                )
            measurement_id += 1

        for icalc_tpl in intermediate_calcs.iteritems():
            calculations.append(icalc_tpl)


        #### activity times by issue tests
        intermediate_calcs = {
            "issues_activity_times": ASADataSet(['group', 'user', 'duration_i2', 'duration_i6'], []),
            "issues_activity_times_sstats": ASADataSet(sstats_headers[:1] + ['issue'] + sstats_headers[1:], []),
            "issues_activity_times_mww_test": ASADataSet(['issue'] + mww_headers, []),
            "issues_activity_times_levene_test": ASADataSet(['issue'] + levene_headers, []),
            "issues_activity_times_ttest_test": ASADataSet(['issue'] + ttest_headers, [])
        }

        issues_activity_times = np.array([np.concatenate((row[0:2], [sum(row[[2,3,5,6]], timedelta())], [sum(row[[4,7,8]], timedelta())])) for row in activity_times_array]).tolist()
        intermediate_calcs["issues_activity_times"].data.extend(issues_activity_times)

        groups_data = self._get_groups_in_seconds(np.array(issues_activity_times), (2, None))
        for idx, name in [(0, "understanding"), (1, "finding")]:
            group1_data = groups_data[0][idx]
            group2_data = groups_data[1][idx]
            intermediate_calcs["issues_activity_times_sstats"].data.extend(
                [(self.groups[0].name, name) + get_simple_stats(group1_data),
                 (self.groups[1].name, name) + get_simple_stats(group2_data)]
            )
            intermediate_calcs["issues_activity_times_mww_test"].data.extend(
                [(name,) + get_mww(group1_data, group2_data)]
            )
            issues_levene_result = get_levene(group1_data, group2_data)
            intermediate_calcs["issues_activity_times_levene_test"].data.extend(
                [(name,) +  issues_levene_result]
            )
            if issues_levene_result[0] > 0.05: # equal variance
                intermediate_calcs["issues_activity_times_ttest_test"].data.extend(
                    [(name,) + get_ttest_equal_var(group1_data, group2_data) + ("equal",)]
                )
            else:
                intermediate_calcs["issues_activity_times_ttest_test"].data.extend(
                    [(name,) + get_ttest_equal_var(group1_data, group2_data) + ("diff",)]
                )

        for icalc_tpl in intermediate_calcs.iteritems():
            calculations.append(icalc_tpl)


        #### totals activity times tests

        total_activity_times = np.array([np.concatenate((row[0:2], [sum(row[2:], timedelta())])) for row in activity_times_array]).tolist()
        calculations.append(("total_activity_times", ASADataSet(['group', 'user', 'duration'], total_activity_times)))

        groups_data = self._get_groups_in_seconds(np.array(total_activity_times), (2, None))
        group1_data = groups_data[0][0]
        group2_data = groups_data[1][0]
        calculations.append(("total_activity_times_sstats", ASADataSet(sstats_headers,
            [(self.groups[0].name,) + get_simple_stats(group1_data), (self.groups[1].name,) + get_simple_stats(group2_data)])))
        calculations.append(("total_activity_times_mww_test", ASADataSet(mww_headers, [get_mww(group1_data, group2_data)])))
        total_levene_result = get_levene(group1_data, group2_data)
        calculations.append(("total_activity_times_levene_test", ASADataSet(levene_headers, [total_levene_result])))
        if total_levene_result[0] > 0.05: # equal variance
            calculations.append(("total_activity_times_ttest_test", ASADataSet(ttest_headers, [get_ttest_equal_var(group1_data, group2_data) + ("equal",)])))
        else:
            calculations.append(("total_activity_times_ttest_test", ASADataSet(ttest_headers, [get_ttest_diff_var(group1_data, group2_data) + ("diff",)])))


        # questionnaires

        questionnaire_questions = ASADataSet(['question_number', 'question'], self.get_questionnaire_questions())
        questionnaire_hypothesis = ASADataSet(['question_number', 'hypothesis'], self.get_questionnaire_hypothesis())
        questionnaire_histogram = ASADataSet(['group', 'question', '1', '2', '3', '4', '5'], [])
        questionnaire_one_answer_per_row = ASADataSet(['group', 'user', 'question', 'answer'], [])
        questionnaire_one_answer_per_column = ASADataSet([], [])
        questionnaire_sstats = ASADataSet(['question'] + sstats_headers, [])
        questionnaire_mww_results = ASADataSet(['question'] + mww_headers, [])
        questionnaire_ttest_results = ASADataSet(['question'] + ttest_headers, [])
        questionnaire_levene_results = ASADataSet(['question'] + levene_headers, [])

        def get_question_and_answers(questionnaire_row):
            question = questionnaire_row[0]
            answers = questionnaire_row[1:-6]
            if type(answers[0]) is float: # discard questions with a non-numeric answer
                answers = [int(answer) if type(answer) is float else answer for answer in answers] # floats become ints
                answers_noned = [a if not a == "" else None for a in answers] # replace missing data values with None
                answers = [a for a in answers if not a == ""] # discard missing data values
                return question, answers, answers_noned
            return question, None, None

        group1_name = self.groups[0].name
        group2_name = self.groups[1].name
        group1_subjects = self.groups[0].get_questionnaire_subjects()
        group1_data = self.groups[0].get_questionnaire_questions_and_answers()
        group2_subjects = self.groups[1].get_questionnaire_subjects()
        group2_data = self.groups[1].get_questionnaire_questions_and_answers()

        questionnaire_one_answer_per_column.headers = ['question'] + group1_subjects + group2_subjects

        if not group1_data is None:
            for i in range(len(group1_data)): # for each question
                question_g1, answers_g1, answers_g1_noned = get_question_and_answers(group1_data[i])
                question_g2, answers_g2, answers_g2_noned = get_question_and_answers(group2_data[i])
                assert question_g1 == question_g2
                if  answers_g1 is None or answers_g2 is None:
                    continue
                for i in range(len(group1_subjects)):
                    if not answers_g1_noned[i] is None:
                        questionnaire_one_answer_per_row.data.append((group1_name, group1_subjects[i], question_g1, answers_g1_noned[i]))
                for i in range(len(group2_subjects)):
                    if not answers_g2_noned[i] is None:
                        questionnaire_one_answer_per_row.data.append((group2_name, group2_subjects[i], question_g2, answers_g2_noned[i]))

                questionnaire_one_answer_per_column.data.append((question_g1,) + tuple(answers_g1_noned + answers_g2_noned))
                questionnaire_histogram.data.append((group1_name, question_g1) + tuple(np.bincount(np.array(answers_g1), minlength=6)[1:]))
                questionnaire_histogram.data.append((group2_name, question_g2) + tuple(np.bincount(np.array(answers_g2), minlength=6)[1:]))
                questionnaire_sstats.data.append((question_g1, group1_name) + get_simple_stats(answers_g1))
                questionnaire_sstats.data.append((question_g2, group2_name) + get_simple_stats(answers_g2))
                questionnaire_mww_results.data.append((question_g1,) + get_mww(answers_g1, answers_g2))
                quest_levene_result = get_levene(answers_g1, answers_g2)
                questionnaire_levene_results.data.append((question_g1,) + quest_levene_result)
                if quest_levene_result[0] > 0.05: # equal variance
                    questionnaire_ttest_results.data.append((question_g1,) + get_ttest_equal_var(answers_g1, answers_g2) + ("equal",))
                else:
                    questionnaire_ttest_results.data.append((question_g1,) + get_ttest_diff_var(answers_g1, answers_g2) + ("diff",))

        calculations.append(("questionnaire_questions", questionnaire_questions))
        calculations.append(("questionnaire_hypothesis", questionnaire_hypothesis))
        calculations.append(("questionnaire_histogram", questionnaire_histogram))
        calculations.append(("questionnaire_one_answer_per_row", questionnaire_one_answer_per_row))
        calculations.append(("questionnaire_one_answer_per_column", questionnaire_one_answer_per_column))
        calculations.append(("questionnaire_sstats", questionnaire_sstats))
        calculations.append(("questionnaire_mww_test", questionnaire_mww_results))
        calculations.append(("questionnaire_levene_test", questionnaire_levene_results))
        calculations.append(("questionnaire_ttest_test", questionnaire_ttest_results))

        stdout.write(" [finished]\n")
        stdout.flush()

        return ASAExperimentCalculations(self, calculations)

class ASAExperimentGroup(object):
    def __init__(self, name, db_dataset, xls_data, date, questionnaire_data):
        """
        Combines the data from the database with that obtained from
        the xls file, to create an instance of ASAExperimentGroup
        """
        self.name = name
        self.users = set()
        self.tasks = set()
        self.timespans = {} # {(user,task): (start_time,end_time)}
        self.task_durations = [] # (user, task, duration)
        self.activity_times = [] # (user, activity_type, duration)
        self.transposed_activity_times = [] # (user, activity1, activity2, activity3, ...)
        self.db_dataset = db_dataset # [[id, resource_type, resource_id, operation, username, time_started, time_ended]]
        self.questionnaire_data = questionnaire_data

        # process XLS data
        group_end_time = max([row[3] for row in xls_data])
        for row in xls_data:
            self.users.add(row[0])
            self.tasks.add(row[1])
            assert(not (row[0], row[1]) in self.timespans) # finding duplicate tasks for the same user means something went wrong...
            if row[4] in ("yes", "partial"): # only account for completed tasks
                self.timespans[(row[0], row[1])] = (
                        "%s %s:00.0" % (date, row[2]),
                        "%s %s:00.0" % (date, row[3])
                    )
                self.task_durations.append((row[0], row[1],
                    parse_date("%s %s:00.0" % (date, row[3])) -
                    parse_date("%s %s:00.0" % (date, row[2]))
                ))
            else:
                if not CONSIDER_ONLY_COMPLETED_TASKS: # make uncompleted tasks take up the rest of the available time
                    if not row[2] == '':
                        self.timespans[(row[0], row[1])] = (
                                "%s %s:00.0" % (date, row[2]),
                                "%s %s:00.0" % (date, group_end_time)
                            )
                        self.task_durations.append((row[0], row[1],
                            parse_date("%s %s:00.0" % (date, group_end_time)) -
                            parse_date("%s %s:00.0" % (date, row[2]))
                        ))

        # Process DB data (needs refactoring)
        stats_wiki =      self.db_dataset.filter_by_resource_type(ResourceTypes.wiki)
        stats_wiki_view = stats_wiki.filter_by_operation("view").aggregate_timedeltas((1,3,4))
        stats_wiki_edit = stats_wiki.filter_by_operation("edit").aggregate_timedeltas((1,3,4))

        stats_search = self.db_dataset.filter_by_resource_type(ResourceTypes.search).aggregate_timedeltas((1,3,4))

        stats_asa_artifact =      self.db_dataset.filter_by_resource_type(ResourceTypes.asa_artifact)
        stats_asa_artifact_view = stats_asa_artifact.filter_by_operation("view").aggregate_timedeltas((1,3,4))
        stats_asa_artifact_edit = stats_asa_artifact.filter_by_operation("edit").aggregate_timedeltas((1,3,4))

        stats_asa_index =  self.db_dataset.filter_by_resource_type(ResourceTypes.asa_index).aggregate_timedeltas((1,3,4))
        stats_asa_search = self.db_dataset.filter_by_resource_type(ResourceTypes.asa_search).aggregate_timedeltas((1,3,4))

        activity_times = []
        for collection, value_type in [
                (stats_wiki_view, Measurements.wiki_view),
                (stats_wiki_edit, Measurements.wiki_edit),
                (stats_search, Measurements.search),
                (stats_asa_artifact_view, Measurements.asa_artifact_view),
                (stats_asa_artifact_edit, Measurements.asa_artifact_edit),
                (stats_asa_index, Measurements.asa_index),
                (stats_asa_search, Measurements.asa_search)]:
            activity_times.extend(collection.delete_columns((0,1)).insert_column(1, "activity", value_type).data)

        self.activity_times.extend(activity_times)
        self.transposed_activity_times.extend(self._transpose_activity_times(activity_times))

    def _transpose_activity_times(self, activity_times):
        def get_duration_for_user_and_activity(user, activity_type):
            for row in activity_times:
                if row[0] == user and row[1] == activity_type:
                    return row[2]
            return timedelta(0)

        blanked_and_ordered_activity_times = []
        import numpy as np
        for user in set(np.array(activity_times)[:,0]): # unique users
            for activity_type in Measurements.to_list():
                blanked_and_ordered_activity_times.append([user, activity_type, get_duration_for_user_and_activity(user, activity_type)])

        transposed_activity_times = [[user] + np.array(list(row)).transpose().tolist()[2:][0] for user,row in groupby(blanked_and_ordered_activity_times, lambda x: x[0])]
        return transposed_activity_times


    def get_times_for_user_and_task(self, username, taskname):
        if not (username, taskname) in self.timespans:
            return None

        # translate taskname to a start and end time
        start_time, end_time = self.timespans[(username, taskname)]

        by_user_and_date = self.db_dataset.filter_by_username(username).filter_by_date_interval(start_time, end_time)

        stats_wiki =      by_user_and_date.filter_by_resource_type(ResourceTypes.wiki)
        stats_wiki_view = stats_wiki.filter_by_operation("view").aggregate_timedeltas((1,3,4))
        stats_wiki_edit = stats_wiki.filter_by_operation("edit").aggregate_timedeltas((1,3,4))

        stats_search = by_user_and_date.filter_by_resource_type(ResourceTypes.search).aggregate_timedeltas((1,3,4))

        stats_asa_artifact =      by_user_and_date.filter_by_resource_type(ResourceTypes.asa_artifact)
        stats_asa_artifact_view = stats_asa_artifact.filter_by_operation("view").aggregate_timedeltas((1,3,4))
        stats_asa_artifact_edit = stats_asa_artifact.filter_by_operation("edit").aggregate_timedeltas((1,3,4))

        stats_asa_index =  by_user_and_date.filter_by_resource_type(ResourceTypes.asa_index).aggregate_timedeltas((1,3,4))
        stats_asa_search = by_user_and_date.filter_by_resource_type(ResourceTypes.asa_search).aggregate_timedeltas((1,3,4))

        assert(len(stats_wiki_view.data) <= 1)
        assert(len(stats_wiki_edit.data) <= 1)
        assert(len(stats_search.data) <= 1)
        assert(len(stats_asa_artifact_view.data) <= 1)
        assert(len(stats_asa_artifact_edit.data) <= 1)
        assert(len(stats_asa_index.data) <= 1)
        assert(len(stats_asa_search.data)<= 1)

        def ensure_not_empty(asa_dataset):
            return asa_dataset.data[0][3] if len(asa_dataset.data) == 1 else None

        return [
            (Measurements.wiki_view,         ensure_not_empty(stats_wiki_view)),
            (Measurements.wiki_edit,         ensure_not_empty(stats_wiki_edit)),
            (Measurements.search,            ensure_not_empty(stats_search)),
            (Measurements.asa_artifact_view, ensure_not_empty(stats_asa_artifact_view)),
            (Measurements.asa_artifact_edit, ensure_not_empty(stats_asa_artifact_edit)),
            (Measurements.asa_index,         ensure_not_empty(stats_asa_index)),
            (Measurements.asa_search,        ensure_not_empty(stats_asa_search))
        ]

    def get_times_for_all_users_and_tasks(self):
        #replaces Nones with 0:00:00
        return [tuple(value if not value is None else timedelta(0) for value in row) for row in self._get_times_for_all_users_and_tasks()]

    def _get_times_for_all_users_and_tasks(self):
        all_times = []
        for user in self.users:
            for task in self.tasks:
                stats = self.get_times_for_user_and_task(user, task)
                if not stats is None:
                    all_times.append((user, task) + tuple(v for k,v in stats))
        return all_times

    def get_task_times(self):
        return self.task_durations

    def get_activity_times(self):
        return self.transposed_activity_times

    def _sum_times(self, times):
        non_nones = self._non_nones(times)
        return sum(non_nones, timedelta()) if len(non_nones) > 0 else None

    def _count_times(self, times):
        return len(self._non_nones(times))

    def _non_nones(self, times):
        return [t for t in times if not t is None]

    def _calc_sum_avg_std(self, values):
        def timedelta_avg(vals):
            return self._sum_times(vals)/self._count_times(vals)

        if self._count_times(values) == 0:
            return (None, None, None)
        total = self._sum_times(values)
        avg = timedelta_avg(values)
        variance = map(lambda x: timedelta(seconds=math.pow(total_seconds(x - avg),2)), self._non_nones(values))
        std = timedelta(seconds=math.sqrt(total_seconds(timedelta_avg(variance))))
        return (total, avg, std)

    def get_questionnaire_subjects(self):
        if self.questionnaire_data is None: return None
        if len(self.questionnaire_data)==0: return []
        return self.questionnaire_data[1][0][1:-6]

    def get_questionnaire_questions_and_answers(self):
        if self.questionnaire_data is None: return None
        if len(self.questionnaire_data)==0: return []
        return self.questionnaire_data[1][1:]


class ASAExperimentCalculations(object):
    def __init__(self, experiment, calculations):
        """
        :param experiment: The experiment that the calculations refer to
        :param calculations: List of tuples with the results of all the executed calculations. These results were calculated in the same order in which they are stored, with some of them being based on previous ones.
        """
        self.experiment = experiment
        self.calculations = calculations

    def as_tuples_list(self):
        """
        Exposes the internal calculations data
        """
        return self.calculations

    def as_dict(self):
        return dict(self.calculations)

    def get_calculation(self, calc_type, sort_key=None):
        import numpy as np
        if sort_key is None:
            return np.array(dict(self.calculations)[calc_type].data)
        else:
            return np.array(sorted(dict(self.calculations)[calc_type].data, key=sort_key)) #,dtype=('a2,a20,f8,f8,f8,f8,f8,f8')

class ASADataSet(object):
    def __init__(self, headers, iterable_data):
        """
            headers: a list of headers for the data
            iterable_data: an interable of tuples
        """
        self.headers = headers
        self.data = iterable_data

    @staticmethod
    def sum_timedeltas(data):
        summed_deltas = timedelta()
        for row in data:
            summed_deltas += parse_date(row[6]) - parse_date(row[5])
        return summed_deltas

    def delete_columns(self, column_range):
        def delete_from_list(data, column_range):
            return [row[0:column_range[0]] + row[column_range[1]+1:] for row in data]
        return ASADataSet(delete_from_list([self.headers], column_range)[0], delete_from_list(self.data, column_range))

    def insert_column(self, index, column_name, default_value):
        def insert_in_list(data, index, default_value):
            return [row[0:index] + (default_value,) + row[index:] for row in data]
        return ASADataSet(insert_in_list([tuple(self.headers)], index, column_name)[0], insert_in_list(self.data, index, default_value))

class ASADBDataSet(ASADataSet):
    def __init__(self, iterable_data):
        super(ASADBDataSet, self).__init__(
            ['id', 'resource_type', 'resource_id', 'operation', 'username', 'time_started', 'time_ended'],
            iterable_data
        )

    def aggregate_timedeltas(self, col_ids, aggr_func=None):
        """
        col_ids is the list of column indices that should be aggregated. The aggregation function
        can be specified, but is otherwise sum(), and always acts over the time columns. Please
        note that index numbers follow this order:
        id, resource_type, resource_id, operation, username, time_started, time_ended
        """
        if aggr_func is None: aggr_func = ASADataSet.sum_timedeltas

        def set_keys(*indices):
            """Returns a function that returns a tuple of key values"""
            def get_keys(seq, indices=indices):
                keys = []
                for i in indices:
                    keys.append(seq[i])
                return tuple(keys)
            return get_keys

        keyfunc = set_keys(*col_ids)
        aggregated = []
        for k,v in groupby(self.data, key=keyfunc):
            aggregated.append(tuple(list(k) + [aggr_func(v)]))
        return ASADataSet(
            ['resource_type', 'operation', 'username', 'durantion'],
            aggregated)

    def filter_by_username(self, username):
        if not type(username) in (list, set):
            username = [username]
        return ASADBDataSet([row for row in self.data if row[4] in username])

    def filter_by_operation(self, operation):
        return ASADBDataSet([row for row in self.data if row[3] == operation])

    def filter_by_resource_type(self, resource_type):
        return ASADBDataSet([row for row in self.data if row[1] == resource_type])

    def filter_by_date_interval(self, start_time, end_time):
        return ASADBDataSet([row for row in self.data if parse_date(start_time) <= parse_date(row[5]) < parse_date(end_time)]) # and parse_date(start_time) < parse_date(row[6]) < parse_date(end_time)
