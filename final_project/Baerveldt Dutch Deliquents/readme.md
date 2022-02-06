DESCRIPTION: This data set is about the evolution of a friendship network and delinquent behavior of pupils in school classes, collected in the Dutch Social Behavior study, a two-wave survey in classrooms (Houtzager and Baerveldt, 1999). These data are from classrooms of the MAVO track, the lower middle level of the Dutch secondary school system, in which the pupils filled in a questionnaire in the 3d and 4th years, with about one year in between.

DATA FORMAT: .csv

DATA: The data files are for 19 schools, numbered h = 1, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23. For each of these values of h, the following files are available:

Network data files (adjacency matrices):

wave 1: N34_h.csv (grade 3)

wave 2: HN34_h.csv (grade 4)

The relation is defined as giving and receiving emotional support: there is a tie from pupil i to pupil j if i says that he/she receives and/or gives emotional support from/to pupil j.

Actor attributes:

CBEh.csv

with the variables, respectively:

gender (1 = boy, 2 = girl);

a measure of delinquent behavior, measured at the first wave; number of minor offences which the respondent states to have committed, transformed by the formula ln(1+x) to correct the skewness of the distribution;

importance of school friends (varying from 1 = very important to 4 = unimportant).

Changing actor behaviour:

cbch.csv

the same measure of delinquent behavior, measured at waves 1 and 2, transformed by ln(1+x), but now also rounded to integer values.

Dyadic covariate:

cbeh.csv

defined as 1 if the pupils have the same ethnic background, and 0 otherwise.
