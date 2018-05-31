import argparse as ap
parser = ap.ArgumentParser(description='delete the blank line and keep sequence length     less than 50')
parser.add_argument('--input1', type=str) 
parser.add_argument('--input2', type=str) 
#parser.add_argument('--output', type=str, default=str + ".less50")
parser.add_argument('--encoding', type = str, default='UTF-8')
args = parser.parse_args()

import sys
reload(sys)
sys.setdefaultencoding(args.encoding)

import codecs as co
f11 = co.open(args.input1, mode = 'r', encoding = args.encoding)
f12 = co.open(args.input2, mode = 'r', encoding = args.encoding)
f21 = co.open(args.input1+".less50", mode = 'w', encoding = args.encoding)
f22 = co.open(args.input2+".less50", mode = 'w', encoding = args.encoding)
fr1 = f11.readline().decode('utf8')
fr2 = f12.readline().decode('utf8')

while fr1 and fr2:
    if len(fr1.split()) > 0 and len(fr1.split()) <= 50 and len(fr2.split()) > 0 and len(fr2.split()) <=50 :
        f21.write(fr1)
        f22.write(fr2)
	fr1 = f11.readline().decode('utf8')
	fr2 = f12.readline().decode('utf8')

f11.close()
f12.close()
f21.close()
f22.close()
