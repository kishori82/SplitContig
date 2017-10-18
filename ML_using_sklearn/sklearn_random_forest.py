from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
	#X = dataset[:,[3,4]]
	X = dataset[:,[4]]
	y = dataset[:,5]

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3)

	for x in [1,4,8,10,15]:
		forest = RandomForestClassifier(n_estimators=x, random_state=2)
		forest.fit(X, y)
		RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
		max_depth=None, max_features='auto', max_leaf_nodes=None,
		min_samples_leaf=1, min_samples_split=2,
		min_weight_fraction_leaf=0.0, n_estimators=5, n_jobs=1,
		oob_score=False, random_state=2, verbose=0, warm_start=False)
		print("Random Forest accuracy on training set for n_estimators=%d is: %f" % (x,forest.score(X_train, y_train)))
		print("Random Forest accuracy on test set for n_estimators=%d is: %f" % (x,forest.score(X_test, y_test)))
	print "="*40

	print X_train.shape, y_train.shape
	print X_test.shape, y_test.shape




if __name__=="__main__":
    createParser()
    main(sys.argv[1:])





