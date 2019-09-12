# example of dict_srcl_tgtl_path
# dict_srcl_tgtl_path = {
#                        'train': {'src': 'train.en', 'tgt': 'train.de'},\
#                        'val': {'src': 'val.en', 'tgt': 'val.de'},\
#                        'test': {'src': 'test.en', 'tgt': 'test.de'}
#                       }

import numpy as np
import re
import tqdm
import pickle

PAD = '<pad>'  # padding token
UNK = '<unk>'  # unknown token
BOS = '<s>'    # beginning of sentence
EOS = '</s>'   # end of sentence


def reindex(l, idx, boolean=False):
    if boolean:
        idx = [i for i, e in enumerate(idx) if e]
    return list(map(lambda i: l[i], idx))


class vocabularyMT:
    def __init__(self, data, max_vocab, constants):

        # initialization attributes and frequencies
        self.data = data
        self.max_vocab = max_vocab
        self.freqs = {}
        self.vocab = []

        # initialization index_to_label and label_to_index
        self.idx2lab = {}
        self.lab2idx = {}
        self.size = 0

        # add constants to vocab
        self.add_vocabs(constants)

        # create the frequencies dict
        self.make_freq()

        # prune the vocabulary to keep only the top max_vocab words
        if len(self.freqs) > self.max_vocab-self.size:
            self.pick_top_freq()
            self.size += self.max_vocab
        else:
            voc_add = list(self.freqs.keys())
            self.vocab.extend(voc_add)
            self.size += len(voc_add)

        # once vocab is set, we need to create mapping between indexes and words (labels) and vice-versa
        self.create_mappings()
        self.size = len(self.idx2lab)

    def make_freq(self):
        for s in self.data:
            for w in s.split(" "):
                if w not in self.freqs.keys():
                    self.freqs[w] = 1
                self.freqs[w] += 1

    def pick_top_freq(self):
        # sort the frequencies
        sorted_freqs = sorted(self.freqs.items(), key=lambda v: v[1])
        # take the top self.max_vocab
        sorted_freqs = sorted_freqs[-(self.max_vocab-self.size):]
        self.vocab.extend(list(list(zip(*sorted_freqs))[0]))

    def create_mappings(self):
        idx = np.arange(self.size)
        self.idx2lab = dict(zip(idx, self.vocab))
        self.lab2idx = dict(zip(self.vocab, idx))

    def add_vocab(self, tok):
        self.vocab.append(tok)
        # self.idx2lab[self.size] = tok
        # self.lab2idx[tok] = self.size
        self.size += 1

    def add_vocabs(self, toks):
        for tok in toks:
            self.add_vocab(tok)

    def lookup(self, tok):
        # if token exists then retrieve value in dict
        try:
            return self.lab2idx[tok]
        # else return unkown token
        except KeyError:
            return self.lab2idx['<unk>']

    def loopkup_idx(self, i):
        # if index exists then retrieve value in dict idx2lab
        try:
            return self.idx2lab[i]
        # else return unkown token
        except KeyError:
            raise 'Index {} not recognized, give an index between 0 and {}'.format(i, self.size)

    def create_embeddings(self, glove_path, charemb_path=None, debug=True):
        # depending on paths given only GloVe (300) or Charemb+GloVe (400)
        if charemb_path:
            emb_dim = 400
        else:
            emb_dim = 300

        # load GloVe
        print("loading GloVe from txt file...")
        f = open(glove_path, 'r')
        GloVe = {}
        for line in tqdm.tqdm_notebook(f):
            line = line.split(' ')
            word = line[0]
            GloVe[word] = np.array([float(val) for val in line[1:]])

        f.close()

        # create the embeddings for a word in vocabulary
        def create_embedding(w):
            # randomly init emb
            emb = np.random.uniform(-0.1, 0.1, size=emb_dim)
            # if GloVe exists then load on indices 0:300
            if w in GloVe.keys():
                emb[:300] = GloVe[w]
            return (w, emb)

        # create all embeddings
        print('Creating all embeddings for src')
        self.embeddings = dict(tqdm.tqdm_notebook(
            map(create_embedding, self.vocab)))

        # delete embeddings loaded
        del GloVe

        # if required, load charemb kazuma
        if charemb_path:
            # if file is txt, then load and pickle
            print("loading Charemb kazuma from txt file...")
            f = open(charemb_path, 'r')
            CharEmb = {}
            for line in tqdm.tqdm_notebook(f):
                line = line.split(' ')
                word = line[0]
                CharEmb[word] = np.array([float(val) for val in line[1:]])

            f.close()

            # define function to lookup the charemb of a word w (string)
            def charemb_get(w):
                # w is a word
                chars = ['#BEGIN#'] + list(w) + ['#END#']
                found = {}
                # do ngrams for n = 2,3,4
                for i in np.arange(2, 5):
                    # create all the ngrams
                    grams = [chars[j:j+i] for j in range(len(chars)-i+1)]
                    # for each ngram lookup if in kazuma if exists
                    for g in grams:
                        g = '{}gram-{}'.format(i, ''.join(g))
                        e = None
                        if g in CharEmb.keys():
                            e = CharEmb[g]
                            if e is not None:
                                found[g] = e
                # if at least on embedding found, then average all the founds (for each of the 100 dimensions)
                if found:
                    emb = sum(found.values())/len(found)
                # if no charemb found then randomly init uniform vector of size 100
                else:
                    emb = np.random.uniform(-0.1, 0.1, size=100)
                return emb

            # modify the embeddings for a word in vocabulary
            def modify_embedding(w):
                self.embeddings[w][300:] = charemb_get(w)

            # modify all embeddings for charemb
            print('Modify all embeddings for src with charemb')
            for w in tqdm.tqdm_notebook(self.vocab):
                modify_embedding(w)

            del CharEmb
        print('Done with vocabulary.')


class PreprocessorMT:

    def __init__(self, dict_src_tgt_path, path_save,
                 vocab_src=None, vocab_tgt=None,
                 max_vocab_src=50000, max_vocab_tgt=500000, max_seq_length=50,
                 glove_path='data/GloVe/glove.840B.300d.txt', charemb_path=None,
                 print_status_s=10000, shuffle_data=True, test=True):

        # init attributes
        self.files = dict_src_tgt_path
        self.path_save = path_save
        self.max_vocab_src, self.max_vocab_tgt = max_vocab_src, max_vocab_tgt
        self.glove_path = glove_path
        self.charemb_path = charemb_path
        self.vocab_src, self.vocab_tgt = vocab_src, vocab_tgt
        self.max_seq_length = max_seq_length
        self.print_status_s = print_status_s
        self.shuffle_data = shuffle_data
        self.test = test

    def preprocess(self, s):
        # strip the '\n' at end of sentence
        s = s.rstrip('\n')
        # turn to lower case
        s = s.lower()
        # add space before all punctuation: .,?!
        s = re.sub(r"([?!.,])", r" \1", s)
        # remove all double spaces and "
        s = re.sub(r" +", " ", s)
        s = re.sub(r'"', '', s)
        return s.strip()

    def filter_max_seq_length(self, key):
        # get length
        len_src = list(map(len, self.data[key]['src']))
        len_tgt = list(map(len, self.data[key]['tgt']))

        def both_less(a, b): return (
            a <= self.max_seq_length) and (b <= self.max_seq_length)
        print('Loop filter max seq length...')
        bool_idx = list(tqdm.tqdm_notebook(map(both_less, len_src, len_tgt)))

        # filter indices
        self.data[key]['src'] = reindex(
            self.data[key]['src'], bool_idx, boolean=True)
        self.data[key]['tgt'] = reindex(
            self.data[key]['tgt'], bool_idx, boolean=True)

        # get length
        len_src = np.max(list(map(len, self.data[key]['src'])))
        len_tgt = np.max(list(map(len, self.data[key]['tgt'])))
        print("After filtering {} set: max length 'src' = {}  /  max length 'tgt' = {}".format(key, len_src, len_tgt))

    def tokenize_sentence_src(self, s):
        # replace the tokens by indexes
        s = list(map(self.vocab_src.lookup, s.split(" ")))

        # add padding
        s.extend([self.vocab_src.lookup('<pad>')]*(self.max_seq_length-len(s)))
        return s

    def tokenize_sentence_tgt(self, s):
        # replace the tokens by indexes
        s = list(map(self.vocab_tgt.lookup, s.split(" ")))

        # add beginning of sentence and end of sentence tokens
        s.insert(0, self.vocab_tgt.lookup('<s>'))
        s.append(self.vocab_tgt.lookup('</s>'))

        # add padding
        s.extend([self.vocab_src.lookup('<pad>')]
                 * (self.max_seq_length+2-len(s)))
        return s

    def tokenize(self, key):
        # init empty dict for that key
        self.data_tok[key] = {}

        # fill in the token for src and tgt
        print('Tokenize src...')
        self.data_tok[key]['src'] = list(tqdm.tqdm_notebook(
            map(self.tokenize_sentence_src, self.data[key]['src'])))
        self.data_tok[key]['src_len'] = list(
            map(lambda x: len(x.split(" ")), self.data[key]['src']))
        print('Tokenize tgt...')
        self.data_tok[key]['tgt'] = list(tqdm.tqdm_notebook(
            map(self.tokenize_sentence_tgt, self.data[key]['tgt'])))
        self.data_tok[key]['tgt_len'] = list(
            map(lambda x: len(x.split(" "))+2, self.data[key]['tgt']))

    def token2sent(self, tokens, mode='src'):
        if mode == 'src':
            def translate(t): return self.vocab_src.idx2lab[t]
        elif mode == 'tgt':
            def translate(t): return self.vocab_tgt.idx2lab[t]
        s = ' '.join(list(map(translate, tokens)))
        return s

    def tokens2sents(self, mat_tokens, mode='src'):
        if mode == 'src':
            def translate(t): return self.vocab_src.idx2lab[t]
        elif mode == 'tgt':
            def translate(t): return self.vocab_tgt.idx2lab[t]

        def tok2sent(tokens):
            return ' '.join(list(map(translate, tokens)))

        return list(map(tok2sent, mat_tokens))

    def get_length_set(self, mode='train'):
        return len(self.data[mode]['src'])

    def process_files(self):
        self.data = {}

        # 1. Training #
        # create the data for train
        print('Start loading training data.')
        self.data['train'] = {}
        for l in self.files['train'].keys():
            print("Starting on {}".format(l))
            path = self.files['train'][l]
            if isinstance(path, list):
                self.data['train'][l] = []
                for p in path:
                    with open(p) as f:
                        self.data['train'][l].extend(
                            [self.preprocess(line) for line in f])
            else:
                with open(path) as f:
                    self.data['train'][l] = [
                        self.preprocess(line) for line in f]

        print(self.data['train'].keys())
        # verify that training sentences for src and tgt have same length
        assert(len(self.data['train']['src'])
               == len(self.data['train']['tgt']))
        print('Done loading training data.\n')

        print('Start filter length training data sentences.')
        # filter sentences that do not have maximum length
        self.filter_max_seq_length('train')
        print('Done filter length training data sentences.\n')

        print('Create vocab src from training.')
        if not self.vocab_src:
            self.vocab_src = vocabularyMT(
                self.data['train']['src'], self.max_vocab_src, ['<pad>', '<unk>'])
        print('Done creating vocab src from training.\n')

        print('Create vocab tgt from training.')
        if not self.vocab_tgt:
            self.vocab_tgt = vocabularyMT(self.data['train']['tgt'], self.max_vocab_tgt, [
                                          '<pad>', '<unk>', '<s>', '</s>'])
        print('Done creating vocab tgt from training.\n')

        # tokenize the sentences (add pad, unk, bos and eos within the data + convert to indices)
        self.data_tok = {}
        self.tokenize('train')

        # 2. Validation and test #
        # loop over the keys in dict (val, test)
        if self.test:
            sets = ['val', 'test']
        else:
            sets = ['val']
        for k in sets:
            self.data[k] = {}
            # loop over the second level of keys in dict (languages)
            for l in self.files[k].keys():
                path = self.files[k][l]
                if isinstance(path, list):
                    self.data['train'][l] = []
                    for p in path:
                        with open(p) as f:
                            self.data[k][l].extend(
                                [self.preprocess(line) for line in f])
                else:
                    with open(path) as f:
                        self.data[k][l] = [self.preprocess(line) for line in f]

            # verify that sentences for src and tgt have same length
            assert(len(self.data[k]['src']) == len(self.data[k]['tgt']))

            # filter sentences that do not have maximum length
            self.filter_max_seq_length(k)

            # tokenize the sentences (add pad, unk, bos and eos within the data + convert to indices)
            self.tokenize(k)

        # 3. Embeddings with Glove and Charemb for source #
        self.vocab_src.create_embeddings(self.glove_path, self.charemb_path)

    def save(self):
        print('Saving the dataset...')
        emb = '_glove'
        if self.charemb_path:
            emb += '_charemb'

        path = self.path_save + '_' + \
            str(self.max_vocab_src)+'_'+str(self.max_seq_length)+emb+'.p'
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        print('Done saving the data')
        print('Saved at {}'.format(path))

    def process_and_save(self):
        print('#####    Processing the files    #####')
        self.process_files()

        print('#####     Saving the dataset     #####')
        self.save()


class dataset():
    def __init__(self, preprocess_obj):
        self.data = preprocess_obj

    # shuffles src, src_len, tgt, tgt_len in data_tok and  src, tgt in data
    def shuffle(self, mode='train'):
        # compute shuffle indexes
        nb_sentences = self.data.get_length_set(mode)
        shuffle_idx = np.random.choice(
            nb_sentences, nb_sentences, replace=False)

        # shuffle in data_tok
        for k in ['src', 'src_len', 'tgt', 'tgt_len']:
            self.data.data_tok[mode][k] = reindex(
                self.data.data_tok[mode][k], shuffle_idx)

        # shuffle in data
        for k in ['src', 'tgt']:
            self.data.data[mode][k] = reindex(
                self.data.data[mode][k], shuffle_idx)

    def get_batch_data(self, i, mode='train', batch_size=64):
        # mode should be in ['train', 'val', 'test']
        batch_sources = self.data.data_tok[mode]['src'][i*batch_size:(i+1)*batch_size]

        batch_sources_len = self.data.data_tok[mode]['src_len'][i*batch_size:(i+1)*batch_size]

        batch_targets = self.data.data_tok[mode]['tgt'][i*batch_size:(i+1)*batch_size]

        batch_targets_len = self.data.data_tok[mode]['tgt_len'][i*batch_size:(i+1)*batch_size]

        # find max len of all targets, then cut to max_len for
        # all targets (to save time for batch processing)
        max_len = np.max(batch_targets_len)
        batch_targets = list(map(lambda x: x[:max_len], batch_targets))

        return batch_sources, batch_sources_len, batch_targets, batch_targets_len

# examples use:
# dataset_MT = dataset(data)
# dataset_MT.shuffle('train')
# x, xlen, y, ylen = dataset_MT.get_batch_data(0, 'train')
