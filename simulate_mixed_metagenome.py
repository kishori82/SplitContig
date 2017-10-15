from optparse import OptionParser
import sys, re, random,csv
from itertools import product
from fastareader import *

usage= sys.argv[0] + """ -i prodigal file -o output  -t prok/euk"""

parser = None
def createParser():
	global parser
	epilog = """
     	this takes fasta file as input and outputs randomly cut seqences 
        python tabular_prodigal_v3_tsv.py -i  fasta -o x -t prok
"""

	epilog = re.sub(r'[ \t\f\v]+',' ', epilog)

    	parser = OptionParser(usage=usage, epilog=epilog)

    	parser.add_option("-i", "--input_fasta", dest="input_file",
                      help='the input prodigal file [REQUIRED]')
	parser.add_option("-o", "--output tabular format", dest="tabular_out",
                      help='the output tabular filename [REQUIRED]')

	parser.add_option("-t", "--type", dest="type", choices=['euk', 'prok'],
                      help='type of organism')

	parser.add_option("-m", "--size", dest="size", type='int',  default =100000000,  help='max size')

def main(argv, errorlogger = None, runstatslogger = None):
	global parser
	(opts, args) = parser.parse_args(argv)
        
        

	fh=open(opts.tabular_out,'a')
        random.seed(45)

        fastareader = FastaReader(opts.input_file)

        sequences = []
        for seq in fastareader:
           sequences.append( (seq.name, seq.sequence))

        indexes = range(0, len(sequences))
        random.shuffle(indexes)

        count = 0
        tot_size = 0
        for s in indexes:
           sequence = sequences[s][1]
           seq_len = len(sequences[s][1])
           i=0 
           while True and tot_size < opts.size:
              length = random.randint(1500, 3000) 
              j =  i +  length
              if j < seq_len:
                 print ">" + opts.type + "_" + str(count)
                 print sequence[i:j]
                 tot_size += len(sequence[i:j])
              else:
                 print ">" + opts.type + "_" + str(count)
                 print sequence[i:seq_len]
                 tot_size += len(sequence[i:seq_len])
                 break

              i=j 
              count += 1

           if tot_size > opts.size:
              break
              

	
if __name__=="__main__":
    createParser()
    main(sys.argv[1:])

