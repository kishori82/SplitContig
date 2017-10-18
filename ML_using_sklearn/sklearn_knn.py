from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
	y = dataset[:,5]

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3)

	training_accuracy = []
	test_accuracy = []
	# try n_neighbors from 1 to 10.
	neighbors_settings = range(1, 11)
	for n_neighbors in neighbors_settings:
		# build the model
		clf = KNeighborsClassifier(n_neighbors=n_neighbors)
		clf.fit(X_train, y_train)
		# record training set accuracy
		training_accuracy.append(clf.score(X_train, y_train))	
		# record generalization accuracy
		test_accuracy.append(clf.score(X_test, y_test))
	for n_neighbors in neighbors_settings:
		print "training accuracy for %.0f no. of neighbors is: %f" %(n_neighbors,training_accuracy[n_neighbors-1])
	print "="*100
	for n_neighbors in neighbors_settings:
		print "test accuracy for %.0f no. of neighbors is: %f" %(n_neighbors,test_accuracy[n_neighbors-1])
	

	print X_train.shape, y_train.shape
	print X_test.shape, y_test.shape




if __name__=="__main__":
    createParser()
    main(sys.argv[1:])










