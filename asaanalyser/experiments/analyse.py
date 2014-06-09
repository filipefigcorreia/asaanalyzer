# -*- coding: utf-8 -*-
import os
import inspect
from asaanalyser.common.util import xls_to_lists
from asaanalyser.common.model import ASAExperiment, ASAExperimentGroup, ASADatabase, FakeASADatabase

def analyse():
    asaanalyser_path = os.path.join(os.path.dirname(inspect.getfile(analyse)), "../")
    experiments_data_path = os.path.join(asaanalyser_path, "../../experiments")

    tasks_times_xls_path = os.path.join(experiments_data_path, "raw-data/tasks_times.xls")
    experiments_metadata_xls_path = os.path.join(experiments_data_path, "raw-data/experiments.xls")

    output_path = os.path.join(asaanalyser_path, "out/experiments/")

    experiments_metadata = xls_to_lists(experiments_metadata_xls_path)
    tasks_times = xls_to_lists(tasks_times_xls_path)
    experiment_questionnaires = {
        "experiment03": xls_to_lists(os.path.join(experiments_data_path, "experiment3/data/questionnaire.xls"))
    }

    def get_dataset(db_filename):
        if db_filename.endswith(".db"):
            return ASADatabase(db_filename).get_full_dataset()
        elif db_filename.endswith(".csv"):
            # TODO: use xls instead of csv
            return FakeASADatabase(db_filename).get_full_dataset()
        else:
            raise Exception("Unknow data source type {0}".format(db_filename[-3:]))


    exp_all_groups = dict([(a[0],a[1:]) for a in experiments_metadata['groups'][1]])
    del experiments_metadata['groups']
    exp_groups = experiments_metadata
    experiments = []
    for exp_name, (_, exp_grps) in exp_groups.iteritems():
        exp_grps = [exp_grp[0] for exp_grp in exp_grps] # converting list of lists to simple list
        groups = []
        grp_questionnaire_hypos = None
        grp_questionnaire_questions = None
        for grp_name, (grp_headers, grp_data) in [(grp_name,_) for grp_name,_ in tasks_times.iteritems() if grp_name in exp_grps]:
            if not grp_name in exp_all_groups:
                print "No metadata found for group '{0}' of the experiment '{1}'. Ignoring it".format(grp_name, exp_name)
                continue
            db_filename = os.path.join(experiments_data_path, "raw-data")
            db_filename = os.path.join(db_filename, exp_all_groups[grp_name][0])
            grp_db_dataset = get_dataset(db_filename) # 0: datasource

            grp_questionnaire_data = None
            if exp_name in experiment_questionnaires:
                grp_questionnaire_data = experiment_questionnaires[exp_name][grp_name]
                grp_questionnaire_hypos = experiment_questionnaires[exp_name][grp_name[0:2] + "_hypothesis"] #TODO: fix ugly hack
                grp_questionnaire_questions = experiment_questionnaires[exp_name][grp_name[0:2] + "_questions"]

            group = ASAExperimentGroup(
                            grp_name,
                            grp_db_dataset,
                            grp_data,
                            exp_all_groups[grp_name][1], # 1: date
                            grp_questionnaire_data
                        )
            groups.append(group)
        experiments.append(ASAExperiment(exp_name, groups, grp_questionnaire_hypos, grp_questionnaire_questions))

    #datasource_filenames = set([(details["datasource"]) for exp,details in exp_all_groups.iteritems()])


    from asaanalyser.common import print_xls_data_as_csv, print_db_data_as_csv, print_results
    for experiment in experiments:
        output_path_csv = os.path.join(output_path, experiment.name + "/csv")
        output_path_tex = os.path.join(output_path, experiment.name + "/tex")
        output_path_pdf = os.path.join(output_path, experiment.name + "/pdf")
        if not os.path.exists(output_path_csv):
           os.makedirs(output_path_csv)
        if not os.path.exists(output_path_pdf):
           os.makedirs(output_path_pdf)

        print_xls_data_as_csv(output_path_csv, [(k,v) for k,v in tasks_times.iteritems() if k in [g.name for g in experiment.groups]])
        print_db_data_as_csv(output_path_csv, experiment)

        print_results(output_path_csv,
                      output_path_tex,
                      output_path_pdf,
                      experiment)

if __name__ == "__main__":
    analyse()
