import unittest
from datetime import timedelta
from model import ASAExperimentGroup, ASAExperiment, Measurements, ASADBDataSet
import numpy as np

class TestAggregation(unittest.TestCase):

    def setUp(self):
        class ASADummyData(object):
            def get_full_dataset(self):
                return ASADBDataSet([
                    ('1',  'wiki',         '/page',         'view', 'p01', '2013-01-01 00:01:00.000', '2013-01-01 00:22:00.000'), #t1 21m 8
                    ('2',  'asa_artifact', '65',            'view', 'p01', '2013-01-01 00:22:00.000', '2013-01-01 00:32:00.000'), #t2 10m
                    ('3',  'wiki',         '/page/subpage', 'view', 'p01', '2013-01-01 00:32:00.000', '2013-01-01 00:35:00.000'), #t2 3m 6
                    ('4',  'asa_artifact', '65',            'view', 'p01', '2013-01-01 00:40:00.000', '2013-01-01 00:44:00.000'), #t3 4m
                    ('5',  'wiki',         '/page/subpage', 'view', 'p01', '2013-01-01 00:44:00.000', '2013-01-01 00:55:00.000'), #t3 11m

                    ('6',  'wiki',         '',              'view', 'p02', '2013-01-01 00:00:00.000', '2013-01-01 00:05:00.000'), #t1 5m 5
                    ('7',  'index',        '',              'view', 'p02', '2013-01-01 00:20:00.000', '2013-01-01 00:21:00.000'), #t2 1m 3
                    ('8',  'index',        '',              'view', 'p02', '2013-01-01 00:21:00.000', '2013-01-01 00:35:00.000'), #t2 14m 3
                    ('9',  'asa_artifact', '41',            'view', 'p02', '2013-01-01 00:45:00.000', '2013-01-01 00:46:00.000'), #t3 1m 7
                    ('10', 'asa_artifact', '2',             'view', 'p02', '2013-01-01 00:46:00.000', '2013-01-01 00:56:00.000'), #t3 10m 7

                    ('11', 'index',        '',              'view', 'p03', '2013-01-01 00:01:00.000', '2013-01-01 00:30:00.000'), #t1 29m 2
                    ]
                )
        self.db_data = ASADummyData()
        self.xls_data = {
            "exp_name": "my_experiment",
            "exp_headers": ["user","task","start", "end", "completed"],
            "exp_data_cg": [
                ("p01", "t1", "00:00", "00:10", "yes"), #10m 600s
                ("p01", "t2", "00:10", "00:40", "yes"), #30m 1800s
                ("p01", "t3", "00:40", "00:55", "yes"), #15m 900s

                ("p02", "t1", "00:00", "00:20", "yes"), #20m 1200s
                ("p02", "t2", "00:20", "00:40", "yes"), #20m 1200s
                ("p02", "t3", "00:40", "00:55", "yes"), #15m 900s

                ("p03", "t1", "00:00", "00:30", "yes"), #30m 1800s
                ("p03", "t2", "00:30", "00:40", "yes"), #10m 600s
                ("p03", "t3", "00:40", "00:55", "yes"), #15m 900s

                # sum -   t1:600,1200,1800 ; t2:1800,1200,600 ; t3:900,900,900
                # means - t1:1200 ; t2:1000 ; t3:900
            ],
            "exp_data_eg": [
                ("p01", "t1", "00:00", "00:10", "yes"),
                ("p01", "t2", "00:10", "00:40", "yes"),
                ("p01", "t3", "00:40", "00:55", "yes"),

                ("p02", "t1", "00:00", "00:20", "yes"),
                ("p02", "t2", "00:20", "00:40", "yes"),
                ("p02", "t3", "00:40", "00:55", "yes"),

                ("p03", "t1", "00:00", "00:30", "yes"), #30m 1800s
                ("p03", "t2", "00:30", "00:40", "yes"), #10m 600s
                ("p03", "t3", "00:40", "01:00", "yes"), #20m 1200s

                # sum -   t1:600,1200,1800 ; t2:1800,1200,600 ; t3:900,900,1200
                # means - t1:1200 ; t2:1200 ; t3:1000
            ]
        }
        self.exp_date = "2013-01-01"

        self.exp_group1 = ASAExperimentGroup(
                        self.xls_data["exp_name"] + "_group1",
                        self.db_data.get_full_dataset(),
                        self.xls_data["exp_data_cg"],
                        self.exp_date,
                        []
                    )

        self.exp_group2 = ASAExperimentGroup(
                        self.xls_data["exp_name"] + "_group2",
                        self.db_data.get_full_dataset(),
                        self.xls_data["exp_data_eg"],
                        self.exp_date,
                        []
                    )

        self.experiment = ASAExperiment("My Experiment", [self.exp_group1, self.exp_group2])


    def test_aggr(self):
        aggr_dataset = self.db_data.get_full_dataset().aggregate_timedeltas((1,3,4))
        self.assertEqual(len(aggr_dataset.data), 6)
        self.assertTrue(('wiki',     'view', 'p01', timedelta(seconds=(21+3+11)*60)) in aggr_dataset.data)
        self.assertTrue(('asa_artifact', 'view', 'p01', timedelta(seconds=(10+4)*60)) in aggr_dataset.data)
        self.assertTrue(('wiki',     'view', 'p02', timedelta(seconds=5*60)) in aggr_dataset.data)
        self.assertTrue(('asa_artifact', 'view', 'p02', timedelta(seconds=(1+10)*60)) in aggr_dataset.data)
        self.assertTrue(('index',    'view', 'p02', timedelta(seconds=(1+14)*60)) in aggr_dataset.data)

    def test_filter(self):
        p02_dataset = self.db_data.get_full_dataset().filter_by_username("p02")
        p01_dataset = self.db_data.get_full_dataset().filter_by_username("p01")
        self.assertEqual(len(p02_dataset.data), 5)
        self.assertEqual(len(p01_dataset.data), 5)

    def test_td_to_str(self):
        from util import td_to_str
        td1 = timedelta(days=1)
        self.assertEqual(td_to_str(td1), "24:00:00")

        td2 = timedelta(seconds=60)
        self.assertEqual(td_to_str(td2), "0:01:00")

        td3 = timedelta(seconds=39.141675)
        self.assertEqual(td_to_str(td3), "0:00:39.141675")

        td4 = timedelta(hours=25, seconds=39.141675)
        self.assertEqual(td_to_str(td4), "25:00:39.141675")

    def test_get_groups(self):
        calcs = self.experiment.run_tests()

        task_times_sstats = calcs.get_calculation("task_times_sstats", lambda x: x[0])
        task_times_sstats_groups = self.experiment.get_groups(task_times_sstats, (4,5), group_column=1)
        task_times_sstats_groups_expected = np.array([
                            np.array([['1200.0', '1200.0', '900.0']], dtype='|S13'),
                            np.array([['1200.0', '1200.0', '1000.0']], dtype='|S13')
                        ])
        self.assertTrue(
            (task_times_sstats_groups == task_times_sstats_groups_expected).all(),
            "Non-matching arrays:\n\n {0}\n\n {1}".format(task_times_sstats_groups, task_times_sstats_groups_expected)
        )

        activity_times_sstats = calcs.get_calculation("activity_times_sstats", lambda x: x[1])
        activity_times_sstats_groups = self.experiment.get_groups(activity_times_sstats, (2,4), group_column=0)
        activity_times_sstats_groups_expected = np.array([
                            np.array([
                                ['1500.0', '2640.0', '0.0', '2400.0'], # g1 asa_artifact_edit
                                ['500.0',  '880.0',  '0.0', '800.0']   # g2 asa_Artifact_view
                            ], dtype='|S17'),
                            np.array([
                                ['1500.0', '2640.0', '0.0', '2400.0'], # g1 asa_artifact_edit
                                ['500.0',  '880.0',  '0.0', '800.0']   # g2 asa_Artifact_view
                            ], dtype='|S17')
                        ])
        self.assertTrue(
            (activity_times_sstats_groups == activity_times_sstats_groups_expected).all(),
            "Non-matching arrays:\n\nGot: \n{0}\n\nExpected: \n{1}".format(activity_times_sstats_groups, activity_times_sstats_groups_expected)
        )

        """
        np.array([
            np.array([
                ['4800.0', '4800.0', '4800.0', '4800.0', '4800.0', '4800.0', '4800.0'],
                ['1600.0', '1600.0', '1600.0', '1600.0', '1600.0', '1600.0', '1600.0'],
            ], dtype='|S17'),
            np.array([
                ['4800.0', '4800.0', '4800.0', '4800.0', '4800.0', '4800.0', '4800.0'],
                ['1600.0', '1600.0', '1600.0', '1600.0', '1600.0', '1600.0', '1600.0'],
            ], dtype='|S17')
        ])
        """

if __name__ == '__main__':
    unittest.main()