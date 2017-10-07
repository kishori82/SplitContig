from optparse import OptionParser
import sys, re
import csv
from itertools import product
usage= sys.argv[0] + """ -i prodigal file -o output  -t prok/euk"""

parser = None
def createParser():
	global parser
	epilog = """
     	this takes prodigal file as input and outputs it in a tabular format 
        python tabular_prodigal_v3_tsv.py -i output/euk_1.fasta.out -o x -t prok
"""

	epilog = re.sub(r'[ \t\f\v]+',' ', epilog)

    	parser = OptionParser(usage=usage, epilog=epilog)

    	parser.add_option("-i", "--input_prodigal_file", dest="input_file",
                      help='the input prodigal file [REQUIRED]')
	parser.add_option("-o", "--output tabular format", dest="tabular_out",
                      help='the output tabular filename [REQUIRED]')

	parser.add_option("-t", "--type", dest="type", choices=['euk', 'prok'],
                      help='type of organism')

def main(argv, errorlogger = None, runstatslogger = None):
	global parser
	(opts, args) = parser.parse_args(argv)

	fh=open(opts.tabular_out,'a')

	#CK--it reads a tsv file and prints only selected tabs.
	commentPATT = re.compile(r'^#')
	removePATT = re.compile(r'[a-z_]=(\S+)')
        line_no=1
	with open(opts.input_file, 'r') as inf:
    	   for line in inf:
             # if comment skip to continue to the next line
             if commentPATT.search(line):
                 continue

             fields = line.strip().split('\t')
             strand  = fields[6]


             last_field = fields[-1]
                
             # split with ; and then skip the first two
             fields = last_field.split(';')[2:]
              
             fields_of_line=[ opts.type, strand]

             # remove the unwanted part
             for field in fields:
                res = removePATT.search(field)
                if res:
                   fields_of_line.append(res.group(1))
             i=0
             for field in fields_of_line:
               if i==0:
                  fh.write(field)
               else:
                  fh.write('\t' + field)
               i +=1 

             fh.write('\n')
             line_no += 1
	fh.close()
	
if __name__=="__main__":
    createParser()
    main(sys.argv[1:])

