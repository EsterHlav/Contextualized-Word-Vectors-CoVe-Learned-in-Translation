# base preprocessing of raw files (.xml->.tok) got
# from https://github.com/salesforce/cove/tree/master/OpenNMT-py 6:17-> 6:44

cd data
mkdir saved_preprocess

# GloVe
echo "Download GloVe zip file..."
mkdir GloVe
cd GloVe
curl -LO "http://nlp.stanford.edu/data/glove.840B.300d.zip"
unzip glove.840B.300d.zip
rm glove.840B.300d.zip
cd ../
echo "Done loading GloVe."

# Kazuma
echo "Download kazuma char embddings..."
mkdir kazuma
cd kazuma
curl -LOk 'http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz'
tar xf jmt_pre-trained_embeddings.tar.gz
rm word.txt jmt_pre-trained_embeddings.tar.gz
cd ../
echo "Done loading kazuma embeddings."

# Download the scripts for moses tokenizer
curl -LO https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/tokenizer.perl
curl -LO https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/lowercase.perl
sed -i '' -e "s/..\/share\/nonbreaking_prefixes//" tokenizer.perl
curl -LO https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.de
curl -LO https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/share/nonbreaking_prefixes/nonbreaking_prefix.en
curl -LO https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl

# Download MT-Medium   #/data
mkdir IWSLT16
curl -LO https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz && tar -xf de-en.tgz -C IWSLT16
echo 'Done loading MT_Medium dataset!\n'
python preprocess_xml.py
echo 'Done preprocessing MT_Medium dataset!\n'

# Download MT_Large    #/data
mkdir wmt17
cd wmt17
curl -LO http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
curl -LO http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
curl -LO http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz
curl -LO http://data.statmt.org/wmt17/translation-task/rapid2016.tgz
curl -LO http://data.statmt.org/wmt17/translation-task/dev.tgz
echo 'Done loading MT_Large dataset!\n'
tar -xzf training-parallel-europarl-v7.tgz
tar -xzf training-parallel-commoncrawl.tgz
tar -xzf training-parallel-nc-v12.tgz
tar -xzf rapid2016.tgz
tar -xzf dev.tgz
mkdir de-en
# mv *de-en* de-en
mv training/*de-en* de-en
mv dev/*deen* de-en
mv dev/*ende* de-en
mv dev/*.de de-en
mv dev/*.en de-en
mv dev/newstest2009*.en*
mv dev/news-test2008*.en*

pip install pycld2
pip install unicodeblock
python ../clean_wmt.py de-en
for l in de; do for f in de-en/*.clean.$l; do perl ../tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; done; done
for l in en; do for f in de-en/*.clean.$l; do perl ../tokenizer.perl -no-escape -l $l -q  < $f > $f.tok; done; done
for l in en de; do for f in de-en/*.clean.$l.tok; do perl ../lowercase.perl < $f > $f.low; done; done
for l in en de; do perl ../tokenizer.perl -no-escape -l $l -q  < de-en/newstest2013.$l > de-en/newstest2013.$l.tok; done
for l in en de; do perl ../lowercase.perl  < de-en/newstest2013.$l.tok > de-en/newstest2013.$l.tok.low; done
for l in en de; do cat de-en/commoncraw*clean.$l.tok.low de-en/europarl*.clean.$l.tok.low de-en/news-commentary*.clean.$l.tok.low de-en/rapid*.clean.$l.tok.low > de-en/train.clean.$l.tok.low; done
echo 'Done preprocessing MT_Large dataset!\n'
