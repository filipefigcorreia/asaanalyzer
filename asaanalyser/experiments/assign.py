# -*- coding: utf-8 -*-
import os
import inspect
from asaanalyser.common.util import xls_to_lists
from groups_assignment import make_assignments

def assign_expertiment3():
    asaanalyser_path = os.path.join(os.path.dirname(inspect.getfile(assign_expertiment3)), "../")
    experiments_data_path = os.path.join(asaanalyser_path, "../../experiments")

    students_grades_xls_path = os.path.join(experiments_data_path, "experiment3/data/participants.xls")
    output_dir = os.path.join(asaanalyser_path, "out/experiments/assignment/experiment3/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    students_grades = xls_to_lists(students_grades_xls_path)
    students = students_grades["students"][1:][0]
    grades = students_grades["grades"][1:][0]

    number_columns_with_header = len([g for g in students_grades["grades"][0] if g])
    grades_trimmed_to_columns = [g[:number_columns_with_header] for g in grades]

    make_assignments(students, grades_trimmed_to_columns, output_dir, username_prefix="lpoo")

if __name__ == "__main__":
    assign_expertiment3()