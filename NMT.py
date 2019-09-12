import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.util import nest
import numpy as np
#from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import nltk
from preprocessing import reindex
import os
import tqdm
import tensorflow.contrib.seq2seq as s2s
import tensorflow.contrib.slim as slim


def sentence_bleu(references, hypothesis, n=4):
    return round(nltk.translate.bleu_score.sentence_bleu(references, hypothesis), n)

def corpus_bleu(references, hypothesis, n=4):
    return round(nltk.translate.bleu_score.corpus_bleu(references, hypothesis), n)

def reset_graph(seed=97):
    """
    Function to reset tf graph with a certain seed.
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def bleu_process_tok(s, mode='pred'):
    s = s.split(" ")
    out = []
    for w in s:
        if w != '<s>' and w != '</s>':
            out.append(w)
        if w == '</s>':
            if mode == 'tgt':
                return [out]
            else:
                return out

    if mode == 'tgt':
        return [out]
    else:
        return out

def BLEU_set(dataset, idx, pred, mode='val', samples=10):
    sents_src = reindex(dataset.data.data[mode]['src'], idx)
    sents_tgt = reindex(dataset.data.data[mode]['tgt'], idx)

    sents_predicted = dataset.data.tokens2sents(mat_tokens=pred, mode='tgt')

    references_tgt = list(map(lambda x: bleu_process_tok(x, 'tgt'), sents_tgt))
    hypothesis = list(map(bleu_process_tok, sents_predicted))

    bleu_overall = corpus_bleu(references_tgt, hypothesis)
    print('Overall BLEU-4 on {} sentences: {:.4f}'.format(len(sents_tgt), bleu_overall))
    print(30*'--')

    # sample sentences to show
    # to_print_idx = np.random.choice(len(sents_src), samples)
    for i in np.arange(samples):
        print('Example n°{}:\n'.format(i+1))
        print('Src sentence:  {}'.format(sents_src[i]))
        print('Tgt sentence:  {}'.format(sents_tgt[i]))
        print('Tgt predicted: {}'.format(sents_predicted[i]))
        print('Bleu score:    {}'.format(sentence_bleu(bleu_process_tok(sents_tgt[i], 'tgt'), bleu_process_tok(sents_predicted[i]))))
        print(30*'--')


class NMT():

    def __init__(self, params):
        # init attributes from the dictionnary of parameters
        for k, v in params.items():
            self.__dict__[k] = v

        self.dict_lab2idx_src = self.dataset.data.vocab_src.lab2idx
        self.dict_idx2lab_src = self.dataset.data.vocab_src.idx2lab
        self.dict_lab2idx_tgt = self.dataset.data.vocab_tgt.lab2idx
        self.dict_idx2lab_tgt = self.dataset.data.vocab_tgt.idx2lab

        self.vocab_size_src = len(self.dict_lab2idx_src)
        self.vocab_size_tgt = len(self.dict_lab2idx_tgt)

    def build_model(self):
        self.create_inputs()
        self.create_embeddings()
        self.create_lookups()
        self.initialize_session()
        self.add_seq2seq()
        print('Graph of model is built.')

    def create_inputs(self):
        self.words_src = tf.placeholder(tf.int32, shape=[None, None], name='words_source')
        self.words_tgt = tf.placeholder(tf.int32, shape=[None, None], name='words_target')
        self.seq_len_src = tf.placeholder(tf.int32, shape=[None], name='sequence_length_source')
        self.seq_len_tgt = tf.placeholder(tf.int32, shape=[None], name='sequence_length_target')
        self.maximum_iterations = tf.reduce_max(self.seq_len_tgt, name='max_dec_len')

    def create_word_embedding(self, embed_name, vocab_size, embed_dim, train_emb=False, init=None):
        """
        Initialize embedding matrix with shape (vocab_size, embed_dim)
        """
        if train_emb:
            if init is not None:
                embedding = tf.get_variable(embed_name,
                                            #shape=[vocab_size, embed_dim],
                                            dtype=tf.float32, initializer=init)
            else:
                embedding = tf.get_variable(embed_name,
                                            shape=[vocab_size, embed_dim],
                                            dtype=tf.float32)
        # not training the embedding
        else:
            if init is not None:
                embedding = tf.constant(init, dtype=tf.float32)

        return embedding

    def create_embeddings(self):
        """
        Creates embeddings for both languages: src and tgt
        """
        self.embedding_src = self.create_word_embedding('src_embedding', self.vocab_size_src, self.embedding_dim, self.train_emb_src, self.init_emb)
        # by default train embedding for target language
        self.embedding_tgt = self.create_word_embedding('tgt_embedding', self.vocab_size_tgt, self.embedding_dim, True)

    def create_lookups(self):
        self.word_embeddings_src = tf.nn.embedding_lookup(self.embedding_src, self.words_src, name='word_embeddings_src', validate_indices=False)
        self.word_embeddings_tgt = tf.nn.embedding_lookup(self.embedding_tgt, self.words_tgt, name='word_embeddings_tgt', validate_indices=False)

    def LSTM_cell(self, lstm_size, keep_probability):
        """
            Creates and LSTM cell with optional dropout
        """
        cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_probability)
        return cell

    def wrap_att(self, dec_cell, lstm_size, enc_output, lengths, alignment_history=False):
        """
        Wrap a decoder cell within an attention cell like in the paper: global Luong attention.
        """
        attention_mechanism = s2s.LuongAttention(num_units=lstm_size,
                                                 memory=enc_output,
                                                 memory_sequence_length=lengths,
                                                 name='LuongAttention')

        # wrapp as a seq2seq AttentionWrapper
        return s2s.AttentionWrapper(cell=dec_cell,
                                    attention_mechanism=attention_mechanism,
                                    attention_layer_size=None,
                                    output_attention=False,
                                    alignment_history=alignment_history)

    def add_seq2seq(self):
        """
        Creates the seq2seq architecture.
        """
        with tf.variable_scope('dynamic_seq2seq', dtype=tf.float32):
            # Encoder
            encoder_outputs, encoder_state = self.build_encoder()

            # Decoder
            logits, sample_id, final_context_state = self.build_decoder(encoder_outputs,
                                                                        encoder_state)
            if self.mode == 'TRAIN':

                # init the loss graph for training
                loss = self.get_loss(logits)
                self.train_loss = loss
                self.word_count = tf.reduce_sum(self.seq_len_src) + tf.reduce_sum(self.seq_len_tgt)
                self.predict_count = tf.reduce_sum(self.seq_len_tgt)
                self.global_step = tf.Variable(0, trainable=False)
                self.lr = tf.train.exponential_decay(self.lr,
                                                    self.global_step,
                                                    decay_steps=self.lr_decay_steps,
                                                    decay_rate=self.lr_decay,
                                                    staircase=True)

                # using Adam optimizer: https://arxiv.org/abs/1412.6980
                opt = tf.train.AdamOptimizer(self.lr)

                # gradient clipping as it can be very useful for NN, especially
                # for deep RNN based architecture
                if self.clip > 0:
                    grads, vs = zip(*opt.compute_gradients(self.train_loss))
                    grads, gnorm = tf.clip_by_global_norm(grads, self.clip)
                    self.train_op = opt.apply_gradients(zip(grads, vs), global_step=self.global_step)
                else:
                    self.train_op = opt.minimize(self.train_loss, global_step=self.global_step)
            # in INFER mode, only need to create graph to ouput
            # NOTE: self.sample_id is already the argmax over the softmax layer, so it outputs
            # the integer associated with the predicted class (predicted word)
            elif self.mode == 'INFER':
                loss = None
                self.infer_logits, _, self.final_context_state, self.sample_id = logits, loss, final_context_state, sample_id
                self.sample_words = self.sample_id

    def build_encoder(self):
        """
        Build the encoder: multi-layers of bidirectional LSTMs
        """
        with tf.variable_scope("encoder"):

            # here we have to create the bidirectional encoder
            # it consists of h_forward and h_backward for every time step that are
            # concatenated together (and same thing for the cell state)
            # Hence:
            # bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
            # bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
            # also because we concatenate them the dim gets doubled, so we start
            # by initializing the cell sizes by self.lstm_size_enc // 2
            fw_cell = self.LSTM_cell(self.lstm_size_enc // 2, self.keep_probability)
            bw_cell = self.LSTM_cell(self.lstm_size_enc // 2, self.keep_probability)

            # in the paper, they call h the output and hidden state, here we differentiate:
            # out_t = output of LSTM cell at time t       -|
            # h_t = hidden state of LSTM cell at time t    |-> all are doubled because of forward/backward
            # c_t = cell state of LSTM cell at time t     -|
            # In tf, we get as output of biLSTM:
            # out_fw = outputs of LSTM cells at each timestep
            # state_fw = (h_forward, c_forward) at last timestep
            # state_bw = (h_backward, c_backward) at last timestep
            for _ in range(self.nb_layers_enc):
                (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cell,
                    cell_bw=bw_cell,
                    inputs=self.word_embeddings_src,
                    sequence_length=self.seq_len_src,
                    dtype=tf.float32)
                encoder_outputs = tf.concat((out_fw, out_bw), -1)

            bi_state_c = tf.concat((state_fw.c, state_bw.c), -1)
            bi_state_h = tf.concat((state_fw.h, state_bw.h), -1)
            # we need to wrap that concatenated states in a LSTMStateTuple to pass
            # as a proper LSTM cell
            bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
            # wrap as tuple to give it to each layers
            encoder_state = tuple([bi_lstm_state] * self.nb_layers_enc)

            # create self.encoder_outputs to access it in graph from CoVe (this is the cove representation after the encoding step) by concatenating with GLoVe representation
            self.encoder_outputs = encoder_outputs

            return encoder_outputs, encoder_state

    def build_decoder(self, encoder_outputs, encoder_state):
        """
        Build the decoder: multi-layers of LSTMs with global attention mechanism.
        """

        sos_id_2 = tf.cast(self.dict_lab2idx_tgt[self.SOS], tf.int32)
        eos_id_2 = tf.cast(self.dict_lab2idx_tgt[self.EOS], tf.int32)

        self.output_layer = Dense(self.vocab_size_tgt, name='output_projection')

        # Decoder.
        with tf.variable_scope("decoder") as decoder_scope:

            cell, decoder_initial_state = self.build_decoder_cell(
                encoder_outputs,
                encoder_state,
                self.seq_len_src)

            # Train
            if self.mode != 'INFER':

                # tf helper for embedding decoder for training (feed back target and not predicted value)
                helper = s2s.TrainingHelper(self.word_embeddings_tgt,
                                                           self.seq_len_tgt)

                # decoder cell
                decoder_cell = s2s.BasicDecoder(cell,
                                                             helper,
                                                             decoder_initial_state,
                                                             output_layer=self.output_layer)

                # Dynamic decoding
                outputs, final_context_state, _ = s2s.dynamic_decode(decoder_cell,
                                                                    maximum_iterations=self.maximum_iterations,
                                                                    swap_memory=False,
                                                                    impute_finished=True,
                                                                    scope=decoder_scope)
                # Ouputs of decoding
                sample_id = outputs.sample_id
                logits = outputs.rnn_output


            else:
                start_tokens = tf.fill([self.batch_size], sos_id_2)
                end_token = eos_id_2

                # tf helper for embedding decoder for inference (feed back predicted value and not target like in training)
                # NOTE: there must be a bug in the tf helper or in the way I am feeding the inputs as the training gets very high accuracy (close to perfect translation) (with TrainingHelper) but the performance drops with  GreedyEmbeddingHelper (at inference time) with inaccurate translations.. My guess is that there is a shift in the targets somewhere and that TrainingHelper feeds to the decoder at time t the target t, so the decoder is learning identity. But then GreedyEmbeddingHelper does not do the same because it feeds the predicted last target, hence since the NN learned identity it repeats the same word like if it did not learn anything.
                # This is essentially the reason why I was not able to move further and finish training CoVe vectors and then reproduce the results.
                helper = s2s.GreedyEmbeddingHelper(self.embedding_tgt,
                                                  start_tokens,
                                                  end_token)

                decoder_cell = s2s.BasicDecoder(cell,
                                                 helper,
                                                 decoder_initial_state,
                                                 output_layer=self.output_layer)

                # Dynamic decoding
                outputs, final_context_state, _ = s2s.dynamic_decode(
                    decoder_cell,
                    maximum_iterations=self.maximum_iterations,
                    impute_finished=False,
                    swap_memory=False,
                    scope=decoder_scope)

                logits = outputs.rnn_output
                sample_id = outputs.sample_id

        self.logits = logits
        self.sample_id = sample_id

        return logits, sample_id, final_context_state

    def build_decoder_cell(self, encoder_outputs, encoder_state,
                           seq_len_src):
        """
        Build the decoder cell used to build the encoder (that's where we add attention)
        """
        memory = encoder_outputs
        batch_size = self.batch_size

        if self.nb_layers_dec is not None:
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.LSTM_cell(self.lstm_size_dec, self.keep_probability) for _ in
                 range(self.nb_layers_dec)])

        else:
            lstm_cell = self.LSTM_cell(self.lstm_size_dec, self.keep_probability)

        # wrap lstm_cell with AttentionWrapper
        cell = self.wrap_att(lstm_cell,
                            self.lstm_size_dec,
                            memory,
                            seq_len_src)

        # initialize the decoder state as the last state of the encoder hence the .clone(encoder_state)
        decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)

        return cell, decoder_initial_state

    def get_loss(self, logits):
        """
        Function to compute loss, used during BPTT
        """
        target_output = self.words_tgt
        max_time = self.maximum_iterations

        # mask sequence loss with seq_length input (only takes into account loss of first l).
        target_weights = tf.sequence_mask(self.seq_len_tgt,
                                          max_time,
                                          dtype=tf.float32,
                                          name='mask')

        loss = s2s.sequence_loss(logits=logits,
                                                targets=target_output,
                                                weights=target_weights,
                                                average_across_timesteps=True,
                                                average_across_batch=True)

        return loss

    def eval_set(self, dataset_MT, mode='train'):
        # computes BLEU-4 score on whole set
        sentences_tgt = []
        sentences_predicted = []
        print("Start computing BLEU-4 score on {} set".format(mode))
        for i in tqdm.tqdm_notebook(np.arange(dataset_MT.data.get_length_set(mode))):
            # get batch of data
            x, x_len, y, y_len = dataset_MT.get_batch_data(i=i, batch_size=self.batch_size, mode=mode)
            #  create dic to feed
            feed_d = {
                self.words_src: x,
                self.seq_len_src: x_len,
                self.words_tgt: y,
                self.seq_len_tgt: y_len,
            }
            # compute output
            s_ids = self.sess.run([self.sample_id], feed_dict=feed_d)
            # get sentences:
            sents_tgt = list(map(bleu_process_tok, dataset_MT.data.tokens2sents(mat_tokens=y, mode='tgt')))
            sents_predicted = list(map(bleu_process_tok, dataset_MT.data.tokens2sents(mat_tokens=s_ids[0], mode='tgt')))
            # append bleu-ready senteces
            sentences_tgt.extend(sents_tgt)
            sentences_predicted.extend(sents_predicted)
        # once all tgt/translations are available, compute the overall BLEU-4 metric
        return sentence_bleu(sentences_predicted, sentences_tgt)

    def eval_train(self, dataset_MT,  mode='train', print_nb=4):
        # eval and print on a small batch
        idxs = np.random.choice(dataset_MT.data.get_length_set(mode), self.batch_size)
        # requires to have batch_size examples
        s_src = reindex(dataset_MT.data.data_tok[mode]['src'], idxs)
        s_tgt = reindex(dataset_MT.data.data_tok[mode]['tgt'], idxs)

        feed_d = {
                self.words_src: s_src,
                self.seq_len_src: reindex(dataset_MT.data.data_tok[mode]['src_len'], idxs),
                self.words_tgt: s_tgt,
                self.seq_len_tgt: reindex(dataset_MT.data.data_tok[mode]['tgt_len'], idxs),
        }

        infer_logits, s_ids = self.sess.run([self.logits, self.sample_id], feed_dict=feed_d)

        sents_src = dataset_MT.data.tokens2sents(mat_tokens=s_src, mode='src')
        sents_tgt = dataset_MT.data.tokens2sents(mat_tokens=s_tgt, mode='tgt')
        sents_predicted = dataset_MT.data.tokens2sents(mat_tokens=s_ids, mode='tgt')

        references_tgt = list(map(lambda x: bleu_process_tok(x, 'tgt'), sents_tgt))
        hypothesis = list(map(bleu_process_tok, sents_predicted))

        bleu_overall = corpus_bleu(references_tgt, hypothesis)
        print ('Overall BLEU-4 on {} sentences: {:.4f}'.format(self.batch_size, bleu_overall))

        # sample sentences to show
        to_print_idx = np.random.choice(self.batch_size, print_nb)
        for i in np.arange(print_nb):
            print(20*'--')
            print('Example n°{}:\n'.format(i+1))
            print('Src sentence:  {}'.format(sents_src[i]))
            print('Tgt sentence:  {}'.format(sents_tgt[i]))
            print('Tgt predicted: {}'.format(sents_predicted[i]))
            print('Bleu score:    {}'.format(sentence_bleu(bleu_process_tok(sents_tgt[i], 'tgt'), bleu_process_tok(sents_predicted[i]))))


    def train(self, dataset_MT=None, restore_path=None, save=True, print_nb=4, print_bleu=100):
        """
        Train the NN using BPTT on a dataset (dataset object).
        Can decide to overwrite the checkpoint or not with 'save'.
        """
        if dataset_MT is None:
            dataset_MT = self.dataset

        self.initialize_session()

        # if checkpoint given, then continue training from this checkpoint
        if restore_path is not None:
            self.restore_session(restore_path)

        best_score = np.inf
        count_early_stopping = 0

        for epoch in range(self.epochs + 1):
            print('-------------------- Epoch {} of {} --------------------'.format(epoch, self.epochs))

            # shuffle the input data before every epoch.
            dataset_MT.shuffle(mode='train')
            score = self.optimize_epoch(dataset_MT, epoch, print_nb=print_nb, print_bleu=print_bleu)

            if score <= best_score:
                count_early_stopping = 0
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)
                if save:
                    self.saver.save(self.sess, self.save_path)
                best_score = score
                print("--- new best score ---\n\n")
            else:
                # warm up epochs for the model
                if epoch > 10:
                    count_early_stopping += 1
                if count_early_stopping >= self.patience:
                    print("- early stopping {} epochs without improvement".format(count_early_stopping))
                    break

    def optimize_epoch(self, dataset_MT, epoch, print_nb, print_bleu):

        batch_size = self.batch_size
        # compute number of batched available (-1 to avoid last batch that does not fit the size of self.batch_size)
        nbatches = (dataset_MT.data.get_length_set(mode='train') + batch_size - 1) // batch_size - 1
        losses = []
        # loop over batches of tha epoch
        for i in np.arange(nbatches):
            # get batch
            x, x_len, y, y_len = dataset_MT.get_batch_data(i, mode='train', batch_size=batch_size)

            # create dictionnary to feed the data
            feed_d = {
                self.words_src: x,
                self.seq_len_src: x_len,
                self.words_tgt: y,
                self.seq_len_tgt: y_len
            }

            _, train_loss = self.sess.run([self.train_op, self.train_loss], feed_dict=feed_d)

            # print loss
            if i % self.print_update == 0 or i == (nbatches - 1):
                print('Iteration: {} of {}\ttrain_loss: {:.4f}'.format(i, nbatches - 1, train_loss))
            # compute BLEU score
            if i % print_bleu == 0 or i == (nbatches - 1):
                self.eval_train(dataset_MT,  mode='val', print_nb=print_nb)

            losses.append(train_loss)

        # average the loss over the epoch
        avg_loss = self.sess.run(tf.reduce_mean(losses))
        print('Average epoch score: {}'.format(avg_loss))

        return avg_loss



    def infer(self, inputs, idxs,  restore_path):
        """
        Inference function, gets a dataset and indexes to predict on that datasetself.
        Restore path is required!
        """
        self.initialize_session()
        self.restore_session(restore_path)

        # create input data for NN
        feed = {
            self.words_src: reindex(inputs['src'], idxs),
            self.seq_len_src: reindex(inputs['src_len'], idxs),
            self.words_tgt: reindex(inputs['tgt'], idxs),
            self.seq_len_tgt: reindex(inputs['tgt_len'], idxs),
        }

        # inference with data
        infer_logits, s_ids = self.sess.run([self.infer_logits, self.sample_words], feed_dict=feed)

        return s_ids

    def CoVe(self, inputs, idxs, restore_path, with_emb=False):
        """
        Create CoVe representation (with or without embeddings: GLoVe/GLoVe_charemb).
        Restore path is required!
        """

        self.initialize_session()
        self.restore_session(restore_path)

        # create input data for NN
        feed = {
            self.words_src: reindex(inputs['src'], idxs),
            self.seq_len_src: reindex(inputs['src_len'], idxs),
            self.words_tgt: reindex(inputs['tgt'], idxs),
            self.seq_len_tgt: reindex(inputs['tgt_len'], idxs),
        }

        # inference with data
        if not with_emb:
            cove_out = self.sess.run([self.encoder_outputs], feed_dict=feed)
        else:
            cove_out, emb = self.sess.run([self.encoder_outputs, self.word_embeddings_src], feed_dict=feed)
            cove_out = np.concatenate((cove_out, emb), axis=-1)
        return cove_out

    def initialize_session(self):
        """
        Init of session when start training
        """
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def restore_session(self, restore_path):
        """
        Restore a session when start training or infering
        """
        print('Restore graph from {}'.format(restore_path))
        self.saver.restore(self.sess, restore_path)
        print('Graph restored!')
