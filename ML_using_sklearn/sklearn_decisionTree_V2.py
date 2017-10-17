from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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
	#X = dataset[:,0:5]
	X = dataset[:,[3,4]]
	#X = dataset[:,[4]]
	y = dataset[:,5]

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3)

	max_depth = [1,5,8,11,15]
	list_testAccuracy = []
	for x in max_depth:
		tree = DecisionTreeClassifier(max_depth=x, random_state=0)
		tree.fit(X_train, y_train)
		print("Decision tree accuracy on training set for max_depth=%d is: %f" % (x,tree.score(X_train, y_train)))
		tup_maxDepth_testAccuracy = (x,tree.score(X_test,y_test))
		list_testAccuracy.append(tup_maxDepth_testAccuracy)
	print '='*40

	for i,j in list_testAccuracy:
		print("Decision tree accuracy on test set for max_depth={} is: {}".format(i,j))
	print '='*40

		
	print X_train.shape, y_train.shape
	print X_test.shape, y_test.shape

if __name__=="__main__":
    createParser()
    main(sys.argv[1:])






