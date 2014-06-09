ASA Analyzer
============

The ASA Analyzer module was developed in the context of my PhD. It uses the Python programming language and the numpy and matplotlib modules respectively to run the statistical tests and plot the bar-charts and boxplots. The module takes two main responsibilities — the random assignment of subjects to the experimental groups and the analysis of data collected during the experiment. 

Random Assignment
-----------------

The module uses python’s pseudo-random number generator to distribute subjects among two experimental groups, and validates the resulting assignment by comparing the grades of the two groups on a set of courses using an independent-samples Mann-Whitney U test. Data about the students and their grades is loaded from a spreadsheet file and the output is a) a comma-separated values (CSV) file with the group assignment, b) a boxplot diagram of the grades and c) latex files with the results of the statistical tests, that were later used to include the results in my dissertation.

Analysis of Collected Data
--------------------------

The module loads data collected during the experiment and stored in the local filesystem as spreadsheet and database files. It then produces CSV files with results of statistical tests, PDF files of charts of the data used in its raw or aggregated forms, and latex files for inclusion of the results in this dissertation. More specifically, the analyses run by this module focus on a) the answers to the background questionnaire, b) the platform activity, c) the task times and d) the answers to the assessment questionnaire.

At its core, the ASAAnalyzer module is composed by the classes depicted in the Figure below. The ASAExperiment class represents an experiment, with its several experimental groups (ASAExperimentGroup) and calculations resulting from a statistical analysis (ASAExperimentCalculations). The run_tests() method processes the data collected for each experimental group (stored in memory as instances of the ASADataSet class) by running the statistical tests and it creates an instance of ASAExperimentCalculations that encapsulates all the results (that are also instances of the ASADataSet class). The ASADBDataSet class represents data that has been loaded directly from a database (i.e., platform activity data, in this case) and provides some behavior specific for that context.

Installing
----------

Beware that numpy requires gcc 4.0. To compile on Snow Leopard you may need to set the following environment variables before installing numpy ...

    export CC=/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/gcc-4.0
    export CXX=/Developer/Platforms/iPhoneOS.platform/Developer/usr/bin/g++-4.0
    export ARCHFLAGS='-arch i386 -arch x86_64'

... but you may need to ensure they are not set before installing matplotlib.
