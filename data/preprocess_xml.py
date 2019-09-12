# tokenization of files for xml processing from https://github.com/salesforce/cove/tree/master/OpenNMT-py
import os
import glob
import subprocess
import xml.etree.ElementTree as ET
from argparse import ArgumentParser

threads = 8
corenlp = False
aggressive = False
tags = ['seg']

def tokenize(f_txt):
    lang = os.path.splitext(f_txt)[1][1:]
    print ("lang: {}".format(lang))
    f_tok = f_txt
    if aggressive:
        f_tok += '.atok'
    elif corenlp and lang == 'en':
        f_tok += '.corenlp'
    else:
        f_tok += '.tok'
    
    print (f_tok, f_txt)
    with open(f_tok, 'w') as fout, open(f_txt, 'r') as fin:
        if aggressive:
            pipe = subprocess.call(['perl', 'tokenizer.perl', '-a', '-q', '-threads', str(threads), '-no-escape', '-l', lang], stdin=fin, stdout=fout)
        elif corenlp and lang=='en':
            pipe = subprocess.call(['python', 'corenlp_tokenize.py', '-input-fn', f_txt, '-output-fn', f_tok])
        else:
            print ("Calling tokenizer.perl...")
            pipe = subprocess.call(['perl', 'tokenizer.perl', '-q', '-threads', str(threads), '-no-escape', '-l', lang], stdin=fin, stdout=fout)
            print ("Done tokenizing: {} > {}".format(f_txt, f_tok))
            #equivalent: perl tokenizer.perl -q -threads 4 -no-escape -l en < data/IWSLT16/de-en/train.de-en.en > data/IWSLT16/de-en/train.de-en.de.tok

path= 'IWSLT16/de-en/'

for f_xml in glob.iglob(os.path.join(path, '*.xml')):
    print(f_xml)
    f_txt = os.path.splitext(f_xml)[0] 
    with open(f_txt, 'w') as fd_txt:
        root = ET.parse(f_xml).getroot()[0]
        for doc in root.findall('doc'):
            for tag in tags:
                for e in doc.findall(tag):
                    fd_txt.write(e.text.strip() + '\n')
    tokenize(f_txt)

xml_tags = ['<url', '<keywords', '<talkid', '<description', '<reviewer', '<translator', '<title', '<speaker']
for f_orig in glob.iglob(os.path.join(path, 'train.tags*')):
    print(f_orig)
    f_txt = f_orig.replace('.tags', '')
    with open(f_txt, 'w') as fd_txt, open(f_orig) as fd_orig:
        for l in fd_orig:
            if not any(tag in l for tag in xml_tags):
                fd_txt.write(l.strip() + '\n')
    tokenize(f_txt)