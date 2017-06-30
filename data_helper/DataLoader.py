import jieba # Not needed for char based dictionary.
import random
import numpy as np

from itertools import chain # For merge two generator
from collections import defaultdict # For counter

class GossipingDataLoader(object):

    def __init__(self, word_segment=False):

        '''
        :param word_segment: ture for word based(use jieba), False for char based.
        '''

        self.word_segment = word_segment

        self.qa_pairs = None
        self.UNKNOWN_TOKEN = 'UNK'

        self.word2index = {self.UNKNOWN_TOKEN:0}
        self.index2word = [self.UNKNOWN_TOKEN]
        self.word_counter = None
        self.num_different_words = 1 # for UNKNOWN_TOKEN

    def load_dataset(self, dataset_path, max_length=None):

        '''
        Load the gossiping dataset and filter out the sentence whose length is larger then max_length.
        
        :param dataset_path: the path to gossiping dataset 
        :param max_length: the length upper bound of q,a pairs. 
        :return: None
        '''

        self.qa_pairs = []

        print("Loading the Gossiping dataset...")

        with open(dataset_path, 'r', encoding='utf-8') as dataset:

            for line in dataset:

                line = line.strip('\n')
                question,answer = line.split('\t')

                if type(max_length) is int:
                    if len(question) > max_length or len(answer) > max_length:
                        continue
                self.qa_pairs.append([question,answer])

    def sample(self, sample_num=5):

        '''
        Randomly sample q,a pairs
        :param sample_num: 
        :return: a list of q,a pairs
        '''
        return random.sample(self.qa_pairs, sample_num)

    def calculate_word_frequency(self):

        '''
        To calculate the frequency of each word/char(depend on doing word segment or not) in the qa pairs.  
        :return: None
        '''

        print("Counting the word frequency...")

        self.word_counter = defaultdict(int)

        for question,answer in self.qa_pairs:

            words_list = None

            if self.word_segment:
                qa_words_generator = chain(jieba.cut(question), jieba.cut(answer))
                words_list = [word for word in qa_words_generator]
            else:
                words_list = question + answer

            for word in words_list:
                self.word_counter[word] += 1

    def build_onehot_encoding(self, unknown_bound=0):

        '''
        :param sentence: 
        :param unknown_bound, target the char whose frequency is not larger unknown_bound as 'UNK'
        :return: None
        '''

        assert self.qa_pairs is not None, "Please load the dataset before building the one-hot encoding."

        if unknown_bound != 0 and self.word_counter is None:
            self.calculate_word_frequency()

        self._clean_encoding_history()
        print("Building the onehot encoding...")

        for question, answer in self.qa_pairs:
            self._build_onehot_encoding(question, unknown_bound)
            self._build_onehot_encoding(answer, unknown_bound)

    def word_to_encoding(self, word):

        '''
        Transform the word to one-hot encoding.
        :return: the encoding
        '''

        encoding = np.zeros(self.num_different_words)
        if word not in self.word2index:
            encoding[self.word2index[self.UNKNOWN_TOKEN]] = 1
        else:
            encoding[self.word2index[word]] = 1

        return encoding

    def encoding_to_word(self, encoding):

        '''
        Transform the one-hot encoding to word.
        :return: the word
        '''

        index = np.argmax(encoding)
        return self.index2word[index]

    def word_to_index(self, word):

        if word not in self.word2index:
            return self.word2index[self.self.UNKNOWN_TOKEN]
        else:
            return self.word2index[word]

    def index_to_word(self, index):

        assert index >= 0 and index < self.num_different_words, \
            "[Index to Word] : Index %d out of range[0,%d]" % (index, self.num_different_words-1)
        return self.index2word[index]

    def sentence_to_index(self, sentence):

        index = []
        words = self._word_segment(sentence)

        for word in words:
            index.append(self.word_to_index(word))
        return index

    def index_to_sentence(self, index):

        sentence = ''

        for idx in index:
            sentence += self.index_to_word(idx)
        return sentence

    def _clean_encoding_history(self):

        self.word2index = {self.UNKNOWN_TOKEN:0}
        self.index2word = [self.UNKNOWN_TOKEN]
        self.num_different_words = 1

    def _word_segment(self, sentence):

        '''
        Return a word list segmented from sentence.
        '''

        if self.word_segment:
            words = [word for word in jieba.cut(sentence)]
        else:
            words = sentence
        return words

    def _build_onehot_encoding(self, sentence, unknown_bound):

        '''
        build the relation between word and index.
        :param sentence: question or answer in the dataset.
        :param unknown_bound, target the char whose frequency is not larger unknown_bound as 'UNK'
        :return: None
        '''

        words = self._word_segment(sentence)

        for word in words:
            if self.word_counter[word] <= unknown_bound:
                word = self.UNKNOWN_TOKEN
            if word not in self.word2index:
                self.word2index[word] = self.num_different_words
                self.index2word.append(word)
                self.num_different_words += 1