from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
from optparse import OptionParser
import sys, re

usage= sys.argv[0] + """ -i csv_dataset_as_input file"""

parser = None
def createParser():
	global parser
	epilog = """
     	This code takes input a csv dataset and outputs predicted accuracy both on training set and test set"""
	epilog = re.sub(r'[ \t\f\v]+',' ', epilog)
	parser = OptionParser(usage=usage, epilog=epilog)
	parser.add_option("-i", "--input_file_dataset", dest="input_file",
                      help='the input dataset file [REQUIRED]')
  

def main(argv, errorlogger = None, runstatslogger = None):
	global parser
	(opts, args) = parser.parse_args(argv)
	# load the CSV file as a numpy matrix
	dataset = np.loadtxt(opts.input_file, delimiter=",")
	# separate the data from the target attributes
	X = dataset[:,0:5]
	#X = dataset[:,[3,4]]
	#X = dataset[:,[4]]
	y = dataset[:,5]
	list_test_Accuracy = []
	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3)
	list_testAccuracy = []
	C_values = [0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,2,5,10,100]

	for x in C_values:
		LinearSVC1 = LinearSVC(C = x).fit(X_train, y_train)
		print "Linear SVC training set score with C %f: %f" % (x, LinearSVC1.score(X_train, y_train))
		test_set_Accuracy = (x,LinearSVC1.score(X_test,y_test))
		list_test_Accuracy.append(test_set_Accuracy)
	print '='*40
	for i,j in list_test_Accuracy:
		print "Linear SVC test set score with C %f: %f" % (i,j)
	print '='*40

		
	print X_train.shape, y_train.shape
	print X_test.shape, y_test.shape

if __name__=="__main__":
    createParser()
    main(sys.argv[1:])





