# Copyright 2018 Abhinav Maurya

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from multiprocessing import Pool, Process
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger
from nltk.tokenize import word_tokenize
from optparse import OptionParser, Option
from os import path
from PIL import Image
from sklearn import ensemble, linear_model, neighbors
from wordcloud import WordCloud, STOPWORDS, get_single_color_func
import concurrent.futures
import copy
import fileinput
import itertools
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import nltk
import numpy as np
import os, sys
import pandas as pd
import pickle
import pywt
import random
import re
import scipy
import scipy.sparse as sp
import seaborn as sns
import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import PolynomialFeatures
import time

class AspectSentimentModel:
    '''
    This code depends on presence of a few files: company_stopwords.txt, positive_words.txt, negative_words.txt, perf_text.csv, company_gender_performance_results.csv, pos_tag_dict.pickle
    '''
    def __init__(self, minibatch_size, fix_aspects, include_null_values, outlier_percentile, separate_pos_tags, result_direc, sample_fraction, train_fraction, l1rate, l2rate, detrate, learningrate):
        self.stopwords = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS | set(' '.join(open('../data/company_stopwords.txt').readlines()).strip().split())
        self.positive_words = set(' '.join(open('../data/positive_words.txt').readlines()).strip().split())
        self.negative_words = set(' '.join(open('../data/negative_words.txt').readlines()).strip().split())

        self.conv_match_threshold = 0.9999
        self.viztol = 0.0
        self.min_word_frequency = 10
        self.max_word_frequency_percentile = 0.99
        self.train_fraction = train_fraction
        self.sample_fraction = sample_fraction
        self.num_cores = 32

        self.l1_regularization_rate = l1rate
        self.l2_regularization_rate = l2rate
        self.det_regularization_rate = detrate
        self.learning_rate = learningrate
        self.minibatch_size = minibatch_size
        self.fix_aspects = fix_aspects
        self.include_null_values = include_null_values

        self.outlier_percentile = outlier_percentile
        self.outlier_train = None
        self.outlier_train_probabilities = None

        self.model_probability_train = []
        self.model_probability_test = []
        self.model_bic_train = []
        self.model_bic_test = []
        self.model_assignment_fraction_match_train = []
        self.model_assignment_fraction_match_test = []

        self.company_values = [] #populated with proprietary corporate values
        self.num_company_values = len(self.company_values)
        self.company_value_to_aspect_dict = {self.company_values[i]:i for i in range(self.num_company_values)}

        self.departments = [] #corporate departments
        self.num_departments = len(self.departments)
        self.department_to_aspect_dict = {self.departments[i]:i for i in range(self.num_departments)}

        self.num_aspects = (self.num_company_values-1)*(self.num_departments-1)

        self.noun_verb_tags = set(['NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        self.adjective_adverb_tags = set(['JJ', 'JJR', 'JJS', 'WRB', 'RB', 'RBR', 'RBS'])
        self.separate_pos_tags = separate_pos_tags

        self.stanford_dir = '/home/amaurya/bin/stanford-postagger/'
        self.jarfile = self.stanford_dir + 'stanford-postagger.jar'
        self.modelfile = self.stanford_dir + 'models/english-bidirectional-distsim.tagger'

        self.result_direc=result_direc
        if not os.path.exists(self.result_direc):
            os.makedirs(self.result_direc)

    def load_company_data(self):
        '''
        Load and cleans Company data
        Loads from text_reviews.csv if it is already present
        Loads from perf_text.csv and company_gender_performance_results.csv and saves result to text_reviews.csv for future reuse
        '''
        # pos tagging: load pos tag dictionary if we want to separate feature set
        self.pos_tag_dict = self.perform_pos_tagging()
        print('done pos tagging and getting word to pos dictionary...')

        if os.path.exists('text_reviews.csv'):
            self.text_reviews = pd.read_csv('text_reviews.csv', sep=',', header=0, na_filter=False, na_values=['n/a', 'na', 'nan']).sample(frac=self.sample_fraction, random_state=666)
            self.text_reviews.comment = self.text_reviews.comment.astype(str)
            # self.text_reviews = self.text_reviews[(self.text_reviews.review_type == 'peer') & (self.text_reviews.cycle == '2016MY')]
            print('done loading reviews...')
            return

        # load text feedbacks and filter unwanted feedbacks
        text_reviews_df = pd.read_csv('../data/perf_text.csv', sep=',', header=0, na_filter=False, na_values=['n/a', 'na', 'nan']).sample(frac=self.sample_fraction, random_state=666)
        # text_reviews_df = text_reviews_df[(text_reviews_df.review_type == 'peer') & (text_reviews_df.cycle == '2016MY')]
        text_reviews_df['answer'] = [x.replace('Other', 'Not Applicable (NA)') for x in text_reviews_df['answer']]
        print('done loading reviews...')

        # load census data and merge with text data
        gender_performance_df = pd.read_csv('../data/company_gender_performance_results.csv', sep='\t', header=0, na_filter=False, na_values=['n/a', 'na', 'nan'])
        text_reviews_df = text_reviews_df.merge(gender_performance_df, left_on='reviewee_id', right_on='employee_id')
        
        # determine employee performance
        # text_reviews_df = text_reviews_df[(text_reviews_df['2016 MY'] != 3.0) & (text_reviews_df['2016 MY'] != 4.0)]
        text_reviews_df['performance'] = np.ceil(pd.to_numeric(text_reviews_df['2016 MY'])/3.0) - 1.0
        print('done loading and merging employee gender and performance information...')

        # determine ratings i.e. observed partitions
        T_or_B = [1.0*(x in set(['T1: Feedback', 'T2: Feedback', 'T3: Feedback', 'T1: Company Value', 'T2: Company Value', 'T3: Company Value'])) for x in text_reviews_df.question]
        # text_reviews_df['ratings'] = 3*np.array(T_or_B) + np.array(text_reviews_df.performance) - 1.0
        text_reviews_df['ratings'] = np.array(T_or_B)

        # merge the comment and company value for each feedback
        text_reviews_df['question_type'] = [x.split(':')[0].strip() for x in text_reviews_df.question]
        text_reviews_df['comment_or_companyvalue'] = [x.split(':')[1].strip() for x in text_reviews_df.question]
        text_reviews_df = text_reviews_df.groupby(['reviewee_id', 'reviewer_id', 'question_type', 'ratings'])
        text_reviews_df = pd.DataFrame(list(text_reviews_df.apply(self.merge_company_value_comment)))
        text_reviews_df = text_reviews_df[(text_reviews_df['company_value'] != 'Validates Inputs')]
    
        # remove any rows that have non-finite or NaN ratings
        text_reviews_df = text_reviews_df[np.isfinite(text_reviews_df.ratings) & np.isfinite(text_reviews_df.performance)]
        print('done merging company values with feedback comments...')

        # text_reviews_df['company_value'] = pd.Categorical(text_reviews_df['company_value'])
        # text_reviews_df['aspects'] = text_reviews_df['company_value'].cat.codes

        # decide whether to keep null values in company value and department fields
        text_reviews_df['value_not_assigned'] = ((text_reviews_df['company_value'] == 'Not Applicable (NA)') | (text_reviews_df['department'] == 'No Mapping'))
        if self.include_null_values:
            self.num_value_not_assigned = len([x for x in text_reviews_df['value_not_assigned'] if x])
        else:
            text_reviews_df = text_reviews_df[(text_reviews_df['company_value'] != 'Not Applicable (NA)') & (text_reviews_df['department'] != 'No Mapping')]
            self.num_value_not_assigned = 0

        # determine aspects to initialize to based on company values
        text_reviews_df['aspects'] = [(self.num_company_values-1)*self.department_to_aspect_dict[text_reviews_df.iloc[i]['department']] + self.company_value_to_aspect_dict[text_reviews_df.iloc[i]['company_value']] \
            if (text_reviews_df.iloc[i]['department'] != 'No Mapping' and text_reviews_df.iloc[i]['company_value'] != 'Not Applicable (NA)') else self.num_aspects \
            for i in range(len(text_reviews_df))]
        if self.include_null_values:
            text_reviews_df['aspects'] = [a if (a<self.num_aspects) else np.random.randint(low=0, high=self.num_aspects) for a in text_reviews_df['aspects']]
        print('done initializing aspects...')

        # randomly shuffle all datapoints
        text_reviews_df = text_reviews_df.sample(frac=1, random_state=666).reset_index(drop=True)
        self.text_reviews = text_reviews_df.copy()
        self.text_reviews.to_csv('text_reviews.csv')
        print('done shuffling the data frame...')

    def prepare_company_data(self):
        '''
        Creates sparse data matrices needed for training the aspect-sentiment model
        '''
        # vectorizer = sklearn.feature_extraction.text.CountVectorizer(binary=False, ngram_range=(1,3), min_df=self.min_word_frequency, max_df=self.max_word_frequency_percentile, stop_words=frozenset(self.stopwords), analyzer='word', lowercase=True, strip_accents='unicode')
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(binary=False, ngram_range=(1,3), min_df=self.min_word_frequency, max_df=self.max_word_frequency_percentile, stop_words=frozenset(self.stopwords), analyzer='word', lowercase=True, strip_accents='unicode')
        text_preprocessor = vectorizer.build_preprocessor()
        self.text_reviews['original_comment'] = [x for x in self.text_reviews['comment']]
        self.text_reviews['comment'] = [text_preprocessor(x) for x in self.text_reviews['comment']]

        # create sentence-level data
        self.text_review_sentences = []
        for i in range(len(self.text_reviews)):
            row = self.text_reviews.iloc[i]
            sentences = row['comment'].split('.')
            employee_id = row['employee_id']
            company_value = row['company_value']
            original_comment = row['original_comment']
            for s in sentences:
                self.text_review_sentences.append([s,  employee_id, company_value, original_comment])
        self.text_review_sentences = pd.DataFrame(self.text_review_sentences, columns=['sentence', 'employee_id', 'company_value', 'original_comment'])
        self.text_review_sentences = self.text_review_sentences[self.text_review_sentences.sentence != '']
        print('total sentences in corpus: ' + str(len(self.text_review_sentences)))

        # split data
        # msk = np.random.rand(len(self.text_reviews)) <= self.train_fraction
        msk = np.asarray([1.0*i/len(self.text_reviews) for i in range(len(self.text_reviews))]) <= self.train_fraction
        self.text_reviews_train = self.text_reviews[msk].copy()
        self.text_reviews_test = self.text_reviews[~msk].copy()
        print('size: ' + str(len(self.text_reviews)))
        print('train size: ' + str(len(self.text_reviews_train)))
        print('test size: ' + str(len(self.text_reviews_test)))

        comment_train = list(self.text_reviews_train.comment)
        comment_test = list(self.text_reviews_test.comment)

        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_cores) as executor:
        #   comment_train = list(executor.map(self.stopword_replace, comment_train, chunksize=1000))
        #   comment_test = list(executor.map(self.stopword_replace, comment_test, chunksize=1000))

        vectorizer = vectorizer.fit(comment_train)
        self.features_train = vectorizer.transform(comment_train).T
        self.features_test = vectorizer.transform(comment_test).T
        self.features_sentences = vectorizer.transform(self.text_review_sentences['sentence']).T
        
        self.ratings_train = np.asarray(self.text_reviews_train.ratings.copy(), dtype=int)
        self.ratings_test = np.asarray(self.text_reviews_test.ratings.copy(), dtype=int)

        self.aspects_train = np.asarray(self.text_reviews_train.aspects.copy(), dtype=int)
        self.aspects_test = np.asarray(self.text_reviews_test.aspects.copy(), dtype=int)

        self.feature_names = vectorizer.get_feature_names()
        self.feature_names_to_indices = {self.feature_names[i]:i for i in range(len(self.feature_names))}
        self.non_unigram_indices = [i for i in range(len(self.feature_names)) if len(self.feature_names[i].split()) > 1]
        print('number of feature', len(self.feature_names))

        self.num_ratings = len(set(self.text_reviews.ratings))
        self.num_words = self.features_train.shape[0]
        self.num_instances_train = self.features_train.shape[1]
        self.num_instances_test = self.features_test.shape[1]
        self.num_instances = self.num_instances_train + self.num_instances_test
        self.num_instances_sentences = self.features_sentences.shape[1]

        # self.features_train.asfptype()
        # self.draw_scree_plot(self.features_train)
        # self.initialize_aspects_using_svd()

        print('number of set aspects', self.num_aspects, len(self.text_reviews.aspects.unique()))
        assert self.num_aspects >= len(self.text_reviews.aspects.unique())
        
        print(self.text_reviews['company_value'].value_counts())
        print(self.text_reviews['department'].value_counts())

        print('num ratings: ', self.num_ratings)
        print('num words: ', self.num_words)
        print('num instances train: ', self.num_instances_train)
        print('num instances test: ', self.num_instances_test)
        print('num instances: ', self.num_instances)
        print('num instances sentences: ', self.num_instances_sentences)
        print('num aspects: ', self.num_aspects)

        # pickle.dump(self.features_train, open('features_train.pkl', 'wb'))
        # pickle.dump(self.features_test, open('features_test.pkl', 'wb'))
        # pickle.dump(self.text_reviews_train.reviewee_id, open('employee_id_train.pkl', 'wb'))
        # pickle.dump(self.text_reviews_test.reviewee_id, open('employee_id_test.pkl', 'wb'))
        # pickle.dump(self.text_reviews_train.performance, open('performance_train.pkl', 'wb'))
        # pickle.dump(self.text_reviews_test.performance, open('performance_test.pkl', 'wb'))
        # open('feature_names.csv', 'w').write(','.join(self.feature_names))

        if self.separate_pos_tags:
            # creating binary feature templates
            self.noun_verb_indicators_train = sp.csc_matrix(self.features_train.shape)
            self.adjective_adverb_indicators_train = sp.csc_matrix(self.features_train.shape)
            self.noun_verb_indicators_test = sp.csc_matrix(self.features_test.shape)
            self.adjective_adverb_indicators_test = sp.csc_matrix(self.features_test.shape)
            self.noun_verb_indicators_sentences = sp.csc_matrix(self.features_sentences.shape)
            self.adjective_adverb_indicators_sentences = sp.csc_matrix(self.features_sentences.shape)

            print('train feature shape: ', self.noun_verb_indicators_train.shape)
            print('test feature shape: ', self.noun_verb_indicators_test.shape)
            print('sentences feature shape: ', self.noun_verb_indicators_sentences.shape)

            # finding noun-verb, adjective-adverb features
            noun_verb_indices = []
            adjective_adverb_indices = []
            for i in range(len(self.feature_names)):
                last_word = self.feature_names[i].split()[-1]
                if last_word in self.pos_tag_dict:
                    last_word_pos_tag = self.pos_tag_dict[last_word]
                    if last_word_pos_tag in self.noun_verb_tags:
                        noun_verb_indices.append(i)
                    if last_word_pos_tag in self.adjective_adverb_tags:
                        adjective_adverb_indices.append(i)
            noun_verb_indices = set(noun_verb_indices)
            adjective_adverb_indices = set(adjective_adverb_indices)

            noun_verb_indicators_train_indptr = [0]
            noun_verb_indicators_train_indices = []
            noun_verb_indicators_train_data = []
            adjective_adverb_indicators_train_indptr = [0]
            adjective_adverb_indicators_train_indices = []
            adjective_adverb_indicators_train_data = []
            for j in range(len(self.features_train.indptr)-1):
                indices = self.features_train.indices[self.features_train.indptr[j]:self.features_train.indptr[j+1]]
                data = self.features_train.data[self.features_train.indptr[j]:self.features_train.indptr[j+1]]
                for i,v in zip(indices, data):
                    if i in noun_verb_indices:
                        noun_verb_indicators_train_indices.append(i)
                        noun_verb_indicators_train_data.append(v)
                    if i in adjective_adverb_indices:
                        adjective_adverb_indicators_train_indices.append(i)
                        adjective_adverb_indicators_train_data.append(v)
                noun_verb_indicators_train_indptr.append(len(noun_verb_indicators_train_indices))
                adjective_adverb_indicators_train_indptr.append(len(adjective_adverb_indicators_train_indices))
            self.noun_verb_indicators_train = sp.csc_matrix((noun_verb_indicators_train_data, noun_verb_indicators_train_indices, noun_verb_indicators_train_indptr), shape=self.features_train.shape)
            self.adjective_adverb_indicators_train = sp.csc_matrix((adjective_adverb_indicators_train_data, adjective_adverb_indicators_train_indices, adjective_adverb_indicators_train_indptr), shape=self.features_train.shape)

            noun_verb_indicators_test_indptr = [0]
            noun_verb_indicators_test_indices = []
            noun_verb_indicators_test_data = []
            adjective_adverb_indicators_test_indptr = [0]
            adjective_adverb_indicators_test_indices = []
            adjective_adverb_indicators_test_data = []
            for j in range(len(self.features_test.indptr)-1):
                indices = self.features_test.indices[self.features_test.indptr[j]:self.features_test.indptr[j+1]]
                data = self.features_test.data[self.features_test.indptr[j]:self.features_test.indptr[j+1]]
                for i,v in zip(indices, data):
                    if i in noun_verb_indices:
                        noun_verb_indicators_test_indices.append(i)
                        noun_verb_indicators_test_data.append(v)
                    if i in adjective_adverb_indices:
                        adjective_adverb_indicators_test_indices.append(i)
                        adjective_adverb_indicators_test_data.append(v)
                noun_verb_indicators_test_indptr.append(len(noun_verb_indicators_test_indices))
                adjective_adverb_indicators_test_indptr.append(len(adjective_adverb_indicators_test_indices))
            self.noun_verb_indicators_test = sp.csc_matrix((noun_verb_indicators_test_data, noun_verb_indicators_test_indices, noun_verb_indicators_test_indptr), shape=self.features_test.shape)
            self.adjective_adverb_indicators_test = sp.csc_matrix((adjective_adverb_indicators_test_data, adjective_adverb_indicators_test_indices, adjective_adverb_indicators_test_indptr), shape=self.features_test.shape)

            noun_verb_indicators_sentences_indptr = [0]
            noun_verb_indicators_sentences_indices = []
            noun_verb_indicators_sentences_data = []
            adjective_adverb_indicators_sentences_indptr = [0]
            adjective_adverb_indicators_sentences_indices = []
            adjective_adverb_indicators_sentences_data = []
            for j in range(len(self.features_sentences.indptr)-1):
                indices = self.features_sentences.indices[self.features_sentences.indptr[j]:self.features_sentences.indptr[j+1]]
                data = self.features_sentences.data[self.features_sentences.indptr[j]:self.features_sentences.indptr[j+1]]
                for i,v in zip(indices, data):
                    if i in noun_verb_indices:
                        noun_verb_indicators_sentences_indices.append(i)
                        noun_verb_indicators_sentences_data.append(v)
                    if i in adjective_adverb_indices:
                        adjective_adverb_indicators_sentences_indices.append(i)
                        adjective_adverb_indicators_sentences_data.append(v)
                noun_verb_indicators_sentences_indptr.append(len(noun_verb_indicators_sentences_indices))
                adjective_adverb_indicators_sentences_indptr.append(len(adjective_adverb_indicators_sentences_indices))
            self.noun_verb_indicators_sentences = sp.csc_matrix((noun_verb_indicators_sentences_data, noun_verb_indicators_sentences_indices, noun_verb_indicators_sentences_indptr), shape=self.features_sentences.shape)
            self.adjective_adverb_indicators_sentences = sp.csc_matrix((adjective_adverb_indicators_sentences_data, adjective_adverb_indicators_sentences_indices, adjective_adverb_indicators_sentences_indptr), shape=self.features_sentences.shape)

    def initialize_model(self):
        '''
        Initializes theta, beta, phi parameters
        Initializes data structures to store probabilities of feedback to aspect and aspect-sentiment
        '''
        positive_word_indices = [i for i in range(len(self.feature_names)) if self.feature_names[i] in self.positive_words]
        negative_word_indices = [i for i in range(len(self.feature_names)) if self.feature_names[i] in self.negative_words]

        self.theta = np.zeros((self.num_aspects, self.num_words))
        self.beta = np.zeros((self.num_ratings, self.num_words))
        self.phi = np.zeros((self.num_ratings, self.num_aspects, self.num_words))

        # self.theta = 1e-2*np.random.randn(self.num_aspects, self.num_words)
        # self.beta = 1e-2*np.random.randn(self.num_ratings, self.num_words)
        # self.phi = 1e-2*np.random.randn(self.num_ratings, self.num_aspects, self.num_words)

        self.beta[:,positive_word_indices] = np.multiply(self.beta[:,positive_word_indices], np.sign(self.beta[:,positive_word_indices]))
        self.beta[:,negative_word_indices] = np.multiply(self.beta[:,negative_word_indices], -np.sign(self.beta[:,negative_word_indices]))

        self.p_af_train = np.zeros((self.num_aspects, self.num_instances_train))
        self.p_raf_train = np.zeros((self.num_ratings, self.num_aspects, self.num_instances_train))
        self.p_af_test = np.zeros((self.num_aspects, self.num_instances_test))
        self.p_raf_test = np.zeros((self.num_ratings, self.num_aspects, self.num_instances_test))

        print('theta shape:' + str(self.theta.shape))
        print('beta shape:' + str(self.beta.shape))
        print('phi shape:' + str(self.phi.shape))

    def learn_model(self, max_iterations=100):
        '''
        Performs minibatch gradient descent to learn parameters, and projects parameters to satisfy constraints
        Assigns aspects to feedbacks, and maintains outliers
        '''
        for iteration in range(max_iterations):
            print('iteration: ' + str(iteration))

            effective_learning_rate = 1.0*self.learning_rate / (iteration+1)
            print('effective learning rate: ' + str(effective_learning_rate))

            if self.outlier_train is not None:
                self.minibatch_indices = random.sample([x for x in range(self.num_instances_train) if not self.outlier_train[x]], self.minibatch_size)
            else:
                self.minibatch_indices = random.sample(range(self.num_instances_train), self.minibatch_size)
            # self.minibatch_indices = np.asarray([i%self.num_instances_train for i in range(iteration*self.minibatch_size, (iteration+1)*self.minibatch_size)])

            self.calculate_probabilities(self.l2_regularization_rate, self.l1_regularization_rate)

            self.update_theta(effective_learning_rate, self.l2_regularization_rate, self.l1_regularization_rate, self.det_regularization_rate)
            self.update_beta(effective_learning_rate, self.l2_regularization_rate, self.l1_regularization_rate, self.det_regularization_rate)
            self.update_phi(effective_learning_rate, self.l2_regularization_rate, self.l1_regularization_rate, self.det_regularization_rate)

            print('theta', np.amin(self.theta), np.amax(self.theta))
            print('beta', np.amin(self.beta), np.amax(self.beta))
            print('phi', np.amin(self.phi), np.amax(self.phi))

            self.project_onto_constraints()
            self.update_assignments(iteration, max_iterations)

            print('train aspects: ', self.aspects_train)
            print('test aspects: ', self.aspects_test)
            print('assignment fraction match', self.model_assignment_fraction_match_train)

            if iteration>0 and iteration%100 == 0:
                self.save_model()

            # if self.model_assignment_fraction_match_train != [] and self.model_assignment_fraction_match_train[-1] > self.conv_match_threshold:
            #   break

    def save_model(self):
        '''
        Save the final trained model parameters and important training statistics
        '''
        self.text_reviews_train = self.text_reviews_train.assign(outlier = self.outlier_train)
        self.text_reviews_train = self.text_reviews_train.assign(outlier_probabilities = self.outlier_train_probabilities)
        self.text_reviews_train.sort_values(by='outlier_probabilities', ascending=False, inplace=True)
        self.text_reviews_train.to_csv(self.result_direc+'text_reviews_train_with_outliers.csv')

        self.outlier_indices = [x for x in range(self.num_instances_train) if self.outlier_train[x]]
        self.outlier_vector = self.features_train[:,self.outlier_indices].sum(axis=1)
        np.savetxt(self.result_direc+'outlier_vector.csv', self.outlier_vector, delimiter=',')

        # self.save_sparse_csr(result_direc+'features_train.csv', self.features_train)
        # self.save_sparse_csr(result_direc+'features_test.csv', self.features_test)

        np.savetxt(self.result_direc+'aspects_train.csv', np.asarray(self.aspects_train, dtype=int), delimiter=',', fmt='%d')
        np.savetxt(self.result_direc+'aspects_test.csv', np.asarray(self.aspects_test, dtype=int), delimiter=',', fmt='%d')

        np.savetxt(self.result_direc+'aspects_init_train.csv', np.asarray(self.text_reviews_train.aspects, dtype=int), delimiter=',', fmt='%d')
        np.savetxt(self.result_direc+'aspects_init_test.csv', np.asarray(self.text_reviews_test.aspects, dtype=int), delimiter=',', fmt='%d')
        
        np.savetxt(self.result_direc+'perplexities_train.csv', np.array(self.model_probability_train), delimiter=',')
        np.savetxt(self.result_direc+'perplexities_test.csv', np.array(self.model_probability_test), delimiter=',')

        np.savetxt(self.result_direc+'bic_train.csv', np.array(self.model_bic_train), delimiter=',')
        np.savetxt(self.result_direc+'bic_test.csv', np.array(self.model_bic_test), delimiter=',')

        np.savetxt(self.result_direc+'assignment_fraction_match_train.csv', np.array(self.model_assignment_fraction_match_train), delimiter=',')
        np.savetxt(self.result_direc+'assignment_fraction_match_test.csv', np.array(self.model_assignment_fraction_match_test), delimiter=',')

        np.savetxt(self.result_direc+'theta.csv', self.theta, delimiter=',')
        np.savetxt(self.result_direc+'beta.csv', self.beta, delimiter=',')
        self.phi.tofile(self.result_direc+'phi.csv', sep=',')
        open(self.result_direc+'feature_names.csv', 'w').write(','.join(self.feature_names))

    def load_company_data_test_1(self):
        comment = 20*['I have worked with Aalok during Project 10 to reduce costs to less than 10\% of GB.', 'Aalok did a great job at reducing excess costs for mobiles.', 'He has an eye for detail, and the sense of frugality was evident in the way he addressed the different cost line items.', 'It is necessary for Aalok to look at the way he addressed cost line items.', 'Mit dem Brief kam neue Hoffnung.', 'Er war nur kurz, enthielt keine Anrede, er war mit gleichgültiger Höflichkeit diktiert worden, ohne Anteilnahme, ohne die Absicht, mir durch eine versteckte, vielleicht unfreiwillige Wendung zu verstehen zu geben, daß meine Sache gut stand.', 'Obwohl ich den Brief mehrmals las, nach Worten suchte, die ich in der ersten Aufregung überlesen zu haben fürchtete, und obwohl all meine Versuche, etwas Gutes für mich herauszulesen, mißlangen, glaubte ich einige Hoffnungen in ihn setzen zu können, denn man lud mich ein, oder empfahl mir, zum Werk herauszukommen und mich vorzustellen.', 'zum denn man Werk und mich zun haben ich inder in ersten']
        ratings = 20*[0,0,0,0,1,1,1,1]
        self.text_reviews = pd.DataFrame({'comment': comment, 'ratings': ratings})

    def load_company_data_test_2(self):
        comment = 20*['I have worked with Aalok during Project 10 to reduce costs to less than 10\% of GB.', 'Aalok did a great job at reducing excess costs for mobiles.', 'He has an eye for detail, and the sense of frugality was evident in the way he addressed the different cost line items.', 'It is necessary for Aalok to look at the way he addressed cost line items.', 'Mit dem Brief kam neue Hoffnung.', 'Er war nur kurz, enthielt keine Anrede, er war mit gleichgültiger Höflichkeit diktiert worden, ohne Anteilnahme, ohne die Absicht, mir durch eine versteckte, vielleicht unfreiwillige Wendung zu verstehen zu geben, daß meine Sache gut stand.', 'Obwohl ich den Brief mehrmals las, nach Worten suchte, die ich in der ersten Aufregung überlesen zu haben fürchtete, und obwohl all meine Versuche, etwas Gutes für mich herauszulesen, mißlangen, glaubte ich einige Hoffnungen in ihn setzen zu können, denn man lud mich ein, oder empfahl mir, zum Werk herauszukommen und mich vorzustellen.', 'zum denn man Werk und mich zun haben ich inder in ersten']
        ratings = 20*[0,1,0,1,0,1,0,1]
        self.text_reviews = pd.DataFrame({'comment': comment, 'ratings': ratings})

    def draw_scree_plot(self, A, k=200, filename='screeplot.pdf'):
        U, S, V = scipy.sparse.linalg.svds(A, k=k, return_singular_vectors=True)
        eigvals = sorted(S**2 / np.cumsum(S)[-1], reverse=True)
        singvals = np.arange(k) + 1

        fig = plt.figure(figsize=(8,5))
        plt.plot(singvals, eigvals, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')
        leg = plt.legend(['Eigenvalues from SVD'], loc='best')
        
        # plt.show()
        pp = PdfPages(self.result_direc+filename)
        pp.savefig()
        pp.close()
        plt.close()

        sns.heatmap(U)
        # plt.show()
        pp = PdfPages(self.result_direc+'U.pdf')
        pp.savefig()
        pp.close()
        plt.close()

        sns.heatmap(V)
        # plt.show()
        pp = PdfPages(self.result_direc+'V.pdf')
        pp.savefig()
        pp.close()
        plt.close()

    def initialize_aspects_using_svd(self):
        U, S, V = scipy.sparse.linalg.svds(self.features_train, k=self.num_aspects, return_singular_vectors=True)
        assert len(np.array(np.argmax(V, axis=0)).flatten()) == self.num_instances_train
        self.text_reviews_train['aspects'] = np.array(np.argmax(V, axis=0)).flatten()

    def merge_company_value_comment(self, group):
        if len(group) == 0 or len(group) > 2:
            return {}
        
        company_value = group[group['comment_or_companyvalue'] == 'Company Value']
        if len(company_value) == 0:
            company_value = 'Not Applicable (NA)'
        else:
            company_value = company_value.loc[company_value.index[0]]['answer']
        
        row_comment = group[group['comment_or_companyvalue'] == 'Feedback']
        if len(row_comment) == 0:
            return {}
        row_comment = row_comment.loc[row_comment.index[0]]
        
        result = dict(row_comment)
        result['company_value'] = company_value
        
        return result

    def parallelize_dataframe_operation(self, df, func):
        num_partitions = self.num_cores*10
        df_split = np.array_split(df, num_partitions)
        pool = Pool(self.num_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    def perform_pos_tagging(self, parallel_pos_tagging=True, pickle_name='pos_tag_dict.pickle'):
        # text_reviews_df = self.text_reviews.copy()
        # try:
        #   pos_tag_dict_pickle = open(pickle_name, 'rb')
        #   pos_tag_dict = pickle.load(pos_tag_dict_pickle)
        #   pos_tag_dict_pickle.close()
        # except:
        #   if parallel_pos_tagging:
        #       text_reviews_df = self.parallelize_dataframe_operation(text_reviews_df, self.perform_partial_pos_tagging)
        #   else:
        #       st = StanfordPOSTagger(model_filename=self.modelfile, path_to_jar=self.jarfile)
        #       text_reviews_df['pos_result'] = text_reviews_df['comment'].apply(lambda x: st.tag(x.split()[:500]))

        #   pos_tag_words = []
        #   for i in range(len(text_reviews_df)):
        #       pos_result = text_reviews_df.iloc[i]['pos_result']
        #       for p in pos_result:
        #           word = p[0].strip().lower()
        #           tag = p[1]
        #           pos_tag_words.append([p[0], p[1]])
        #   pos_tag_df = pd.DataFrame(pos_tag_words, columns=['word', 'tag'])
        #   pos_tag_df = pos_tag_df.groupby('word').apply(pd.DataFrame.mode)
        #   pos_tag_df = pos_tag_df[['word', 'tag']]
        #   pos_tag_dict = {pos_tag_df.iloc[i]['word']:pos_tag_df.iloc[i]['tag'] for i in range(len(pos_tag_df)) if pos_tag_df.iloc[i]['tag'] != np.nan}

        #   pos_tag_dict_pickle = open('pos_tag_dict.pickle', 'wb')
        #   pickle.dump(pos_tag_dict, pos_tag_dict_pickle)
        #   pos_tag_dict_pickle.close()

        pos_tag_dict_pickle = open(pickle_name, 'rb')
        pos_tag_dict = pickle.load(pos_tag_dict_pickle)
        pos_tag_dict_pickle.close()
        return pos_tag_dict

    def perform_partial_pos_tagging(self, data):
        st = StanfordPOSTagger(model_filename=self.modelfile, path_to_jar=self.jarfile)
        data['pos_result'] = data['comment'].apply(lambda x: st.tag(x.split()[:500]))
        return data

    def stopword_replace(self, text):
        pattern = re.compile('|'.join(['(\W+)' + x.lower() + '(\W+)' for x in self.stopwords]))
        result = pattern.sub(lambda x: ' ', text.lower())
        return result

    def logsumexp_list(self, l):
        acc = 0.0
        for e in l:
            acc = np.logaddexp(acc, e)
        return acc

    def normalize_safe(self, arr, axis=0):
        result = copy.deepcopy(arr)

        if axis==0:
            for i in range(arr.shape[1]):
                sum_column = -np.inf
                for j in range(arr.shape[0]):
                    sum_column = np.logaddexp(sum_column, arr[j,i])
                result[:,i] -= sum_column
        elif axis==1:
            for i in range(arr.shape[0]):
                sum_row = -np.inf
                for j in range(arr.shape[1]):
                    sum_row = np.logaddexp(sum_row, arr[i,j])
                result[i,:] -= sum_row

        return result

    def project_onto_constraints(self, mode='absolute_sum'):
        if mode == 'absolute_sum':
            theta_sum = self.theta.sum(axis=1)
            for i in range(self.theta.shape[0]):
                self.theta[i,:] /= np.abs(theta_sum[i])

            beta_sum = self.beta.sum(axis=1)
            for i in range(self.beta.shape[0]):
                self.beta[i,:] /= np.abs(beta_sum[i])

            # phi_sum = self.phi.sum(axis=(1,2))
            # for i in range(self.phi.shape[0]):
            #   self.phi[i,:,:] /= np.abs(phi_sum[i])
        elif mode == 'sum':
            theta_sum = self.theta.sum(axis=1)
            for i in range(self.theta.shape[0]):
                self.theta[i,:] /= theta_sum[i]

            beta_sum = self.beta.sum(axis=1)
            for i in range(self.beta.shape[0]):
                self.beta[i,:] /= beta_sum[i]

            # phi_sum = self.phi.sum(axis=(1,2))
            # for i in range(self.phi.shape[0]):
            #   self.phi[i,:,:] /= phi_sum[i]
        elif mode == 'l1':
            theta_sum = np.linalg.norm(self.theta, axis=1)
            for i in range(self.theta.shape[0]):
                self.theta[i,:] /= theta_sum[i]

            beta_sum = np.linalg.norm(self.beta, axis=1)
            for i in range(self.beta.shape[0]):
                self.beta[i,:] /= beta_sum[i]

            # phi_sum = np.linalg.norm(self.phi, axis=(1,2))
            # for i in range(self.phi.shape[0]):
            #   self.phi[i,:,:] /= phi_sum[i]

    def update_theta(self, learning_rate, l2_regularization_rate, l1_regularization_rate, det_regularization_rate):
        grad_theta = np.zeros(self.theta.shape)
        I = self.minibatch_indices

        for a in range(self.num_aspects):
            print('a', a)
            # actual_counts = self.features[self.aspects==a,:].sum(axis=0)
            # expected_counts = np.zeros(actual_counts.shape)
            # for f in range(self.num_instances):
            #   expected_counts += np.exp(p_af[a,f])*self.features[f,:]
            # grad_theta[a,:] = actual_counts - expected_counts

            weights = np.array(1.0*(self.aspects_train[I]==a)).flatten()
            if self.separate_pos_tags:
                actual_counts = self.noun_verb_indicators_train[:,I].dot(weights)
            else:
                actual_counts = self.features_train[:,I].dot(weights)

            weights = np.array(np.exp(self.p_af_train[a,I])).flatten()
            if self.separate_pos_tags:
                expected_counts = self.noun_verb_indicators_train[:,I].dot(weights)
            else:
                expected_counts = self.features_train[:,I].dot(weights)

            if np.isfinite(actual_counts).all() and np.isfinite(expected_counts).all():
                grad_theta[a,:] = actual_counts - expected_counts

        self.theta = (1.0 - learning_rate*l2_regularization_rate)*self.theta + learning_rate*grad_theta/self.minibatch_size
        self.theta = self.theta + learning_rate*det_regularization_rate*np.dot( np.linalg.pinv(np.dot(self.theta, self.theta.T)), self.theta )
        self.theta = pywt.threshold(self.theta, learning_rate*l1_regularization_rate, 'soft')

    def update_beta(self, learning_rate, l2_regularization_rate, l1_regularization_rate, det_regularization_rate):
        grad_beta = np.zeros(self.beta.shape)
        I = self.minibatch_indices

        for r in range(self.num_ratings):
            print('r', r)
            # actual_counts = self.features[self.ratings==r,:].sum(axis=0)
            # expected_counts = np.zeros(actual_counts.shape)
            # for f in range(self.num_instances):
            #   a = self.aspects[f]
            #   expected_counts += np.exp(p_raf[r,a,f])*self.features[f,:]
            # grad_beta[r,:] = actual_counts - expected_counts

            weights = np.array(1.0*(self.ratings_train[I]==r))
            if self.separate_pos_tags:
                actual_counts = self.adjective_adverb_indicators_train[:,I].dot(weights)
            else:
                actual_counts = self.features_train[:,I].dot(weights)

            weights = np.array([np.exp(self.p_raf_train[r, self.aspects_train[f], f]) for f in I])
            if self.separate_pos_tags:
                expected_counts = self.adjective_adverb_indicators_train[:,I].dot(weights)
            else:
                expected_counts = self.features_train[:,I].dot(weights)

            if np.isfinite(actual_counts).all() and np.isfinite(expected_counts).all():
                grad_beta[r,:] = actual_counts - expected_counts

        self.beta = (1.0 - learning_rate*l2_regularization_rate)*self.beta + learning_rate*grad_beta/self.minibatch_size
        self.beta = self.beta + learning_rate*det_regularization_rate*np.dot( np.linalg.pinv(np.dot(self.beta, self.beta.T)), self.beta )
        self.beta = pywt.threshold(self.beta, learning_rate*l1_regularization_rate, 'soft')

    def update_phi(self, learning_rate, l2_regularization_rate, l1_regularization_rate, det_regularization_rate):
        grad_phi = np.zeros(self.phi.shape)
        I = self.minibatch_indices

        for r in range(self.num_ratings):
            for a in range(self.num_aspects):
                print('r', r, 'a', a)
                # actual_counts = self.features[(self.aspects==a) & (self.ratings==r),:].sum(axis=0)
                # expected_counts = np.zeros(actual_counts.shape)
                # for f in range(self.num_instances):
                #   if a != self.aspects[f]:
                #       continue
                #   expected_counts += np.exp(p_raf[r,a,f])*self.features[f,:]
                # grad_phi[r,:] = actual_counts - expected_counts

                weights = np.array(1.0*((self.aspects_train[I]==a) & (self.ratings_train[I]==r)))
                if self.separate_pos_tags:
                    actual_counts = self.adjective_adverb_indicators_train[:,I].dot(weights)
                else:
                    actual_counts = self.features_train[:,I].dot(weights)

                weights = np.array([np.exp(self.p_raf_train[r, a, f])*(a == self.aspects_train[f]) for f in I])
                if self.separate_pos_tags:
                    expected_counts = self.adjective_adverb_indicators_train[:,I].dot(weights)
                else:
                    expected_counts = self.features_train[:,I].dot(weights)

                if np.isfinite(actual_counts).all() and np.isfinite(expected_counts).all():
                    grad_phi[r,a,:] = actual_counts - expected_counts

        phi_flattened = self.phi.reshape((self.num_ratings*self.num_aspects, self.num_words))
        reg_flattened = np.dot( np.linalg.pinv(np.dot(phi_flattened, phi_flattened.T)), phi_flattened )
        reg = reg_flattened.reshape((self.num_ratings, self.num_aspects, self.num_words))

        reg = np.zeros(self.phi.shape)
        for r in range(self.num_ratings):
            mat = self.phi[r,:,:]
            reg[r,:,:] = np.dot( np.linalg.pinv(np.dot(mat, mat.T)), mat )

        self.phi = (1.0 - learning_rate*l2_regularization_rate)*self.phi + learning_rate*grad_phi/self.minibatch_size
        self.phi = self.phi + learning_rate*det_regularization_rate*reg
        self.phi = pywt.threshold(self.phi, learning_rate*l1_regularization_rate, 'soft')

    def calculate_probabilities(self, l2_regularization_rate, l1_regularization_rate, minibatch=True, sentences=False):
        if minibatch:
            I = self.minibatch_indices
        else:
            I = np.array([x for x in range(self.num_instances_train)])
        l1_regularization = l1_regularization_rate*(np.linalg.norm(self.theta.flatten(), 1) + np.linalg.norm(self.beta.flatten(), 1) + np.linalg.norm(self.phi.flatten(), 1))
        l2_regularization = l2_regularization_rate*(np.linalg.norm(self.theta.flatten(), 2) + np.linalg.norm(self.beta.flatten(), 2) + np.linalg.norm(self.phi.flatten(), 2))

        # train data stuff
        p_af_train = np.zeros((self.num_aspects, len(I)))
        p_raf_train = np.zeros((self.num_ratings, self.num_aspects, len(I)))

        if self.separate_pos_tags:
            p_af_train = self.noun_verb_indicators_train[:,I].T.dot(self.theta.T).T
        else:
            p_af_train = self.features_train[:,I].T.dot(self.theta.T).T
        p_af_train = self.normalize_safe(p_af_train, axis=0)

        for r in range(self.num_ratings):
            if self.separate_pos_tags:
                p_raf_train[r,:,:] += self.adjective_adverb_indicators_train[:,I].T.dot(self.phi[r,:,:].T).T
            else:
                p_raf_train[r,:,:] += self.features_train[:,I].T.dot(self.phi[r,:,:].T).T
        
        for a in range(self.num_aspects):
            if self.separate_pos_tags:
                p_raf_train[:,a,:] += self.adjective_adverb_indicators_train[:,I].T.dot(self.beta.T).T
            else:
                p_raf_train[:,a,:] += self.features_train[:,I].T.dot(self.beta.T).T

        p_raf_train = p_raf_train.reshape((self.num_ratings, self.num_aspects*len(I)))
        p_raf_train = self.normalize_safe(p_raf_train, axis=0)
        p_raf_train = p_raf_train.reshape((self.num_ratings, self.num_aspects, len(I)))

        # train perplexity
        model_probability_train = 0.0
        for f in range(len(I)):
            r = self.ratings_train[I[f]]
            for a in range(self.num_aspects):
                model_probability_train += (p_af_train[a,f] + p_raf_train[r,a,f])

        model_bic_train = -2*model_probability_train + (self.theta.size + self.beta.size + self.phi.size)*np.log(len(I))
        self.model_bic_train.append(model_bic_train)

        model_probability_train = -model_probability_train + l1_regularization + l2_regularization
        self.model_probability_train.append(model_probability_train)

        # test data stuff
        p_af_test = np.zeros((self.num_aspects, self.num_instances_test))
        p_raf_test = np.zeros((self.num_ratings, self.num_aspects, self.num_instances_test))

        if self.separate_pos_tags:
            p_af_test = self.noun_verb_indicators_test.T.dot(self.theta.T).T
        else:
            p_af_test = self.features_test.T.dot(self.theta.T).T
        p_af_test = self.normalize_safe(p_af_test, axis=0)

        for r in range(self.num_ratings):
            if self.separate_pos_tags:
                p_raf_test[r,:,:] += self.adjective_adverb_indicators_test.T.dot(self.phi[r,:,:].T).T
            else:
                p_raf_test[r,:,:] += self.features_test.T.dot(self.phi[r,:,:].T).T
        
        for a in range(self.num_aspects):
            if self.separate_pos_tags:
                p_raf_test[:,a,:] += self.adjective_adverb_indicators_test.T.dot(self.beta.T).T
            else:
                p_raf_test[:,a,:] += self.features_test.T.dot(self.beta.T).T

        p_raf_test = p_raf_test.reshape((self.num_ratings, self.num_aspects*self.num_instances_test))
        p_raf_test = self.normalize_safe(p_raf_test, axis=0)
        p_raf_test = p_raf_test.reshape((self.num_ratings, self.num_aspects, self.num_instances_test))

        # test perplexity
        model_probability_test = 0.0
        for f in range(self.num_instances_test):
            r = self.ratings_test[f]
            for a in range(self.num_aspects):
                model_probability_test += (p_af_test[a,f] + p_raf_test[r,a,f])

        model_bic_test = -2*model_probability_test + (self.theta.size + self.beta.size + self.phi.size)*np.log(self.num_instances_test)
        self.model_bic_test.append(model_bic_test)

        model_probability_test = -model_probability_test + l1_regularization + l2_regularization
        self.model_probability_test.append(model_probability_test)

        if sentences:
            # sentences stuff
            p_af_sentences = np.zeros((self.num_aspects, self.num_instances_sentences))
            p_raf_sentences = np.zeros((self.num_ratings, self.num_aspects, self.num_instances_sentences))

            if self.separate_pos_tags:
                p_af_sentences = self.noun_verb_indicators_sentences.T.dot(self.theta.T).T
            else:
                p_af_sentences = self.features_sentences.T.dot(self.theta.T).T
            p_af_sentences = self.normalize_safe(p_af_sentences, axis=0)

            for r in range(self.num_ratings):
                if self.separate_pos_tags:
                    p_raf_sentences[r,:,:] += self.adjective_adverb_indicators_sentences.T.dot(self.phi[r,:,:].T).T
                else:
                    p_raf_sentences[r,:,:] += self.features_sentences.T.dot(self.phi[r,:,:].T).T
            
            for a in range(self.num_aspects):
                if self.separate_pos_tags:
                    p_raf_sentences[:,a,:] += self.adjective_adverb_indicators_sentences.T.dot(self.beta.T).T
                else:
                    p_raf_sentences[:,a,:] += self.features_sentences.T.dot(self.beta.T).T

            p_raf_sentences = p_raf_sentences.reshape((self.num_ratings, self.num_aspects*self.num_instances_sentences))
            p_raf_sentences = self.normalize_safe(p_raf_sentences, axis=0)
            p_raf_sentences = p_raf_sentences.reshape((self.num_ratings, self.num_aspects, self.num_instances_sentences))

        # print perplexities
        print('train perplexities: ', self.model_probability_train)
        print('test perplexities: ', self.model_probability_test)
        print('train bic: ', self.model_bic_train)
        print('test bic: ', self.model_bic_test)
        print('train assignment fraction match: ', self.model_assignment_fraction_match_train)
        print('test assignment fraction match: ', self.model_assignment_fraction_match_test)

        # assign to be reused
        self.p_af_train[:,I] = p_af_train
        self.p_raf_train[:,:,I] = p_raf_train
        self.p_af_test[:,np.array([x for x in range(self.num_instances_test)])] = p_af_test
        self.p_raf_test[:,:,np.array([x for x in range(self.num_instances_test)])] = p_raf_test
        if sentences:
            self.p_af_sentences = p_af_sentences
            self.p_raf_sentences = p_raf_sentences

    def update_assignments(self, iteration, max_iterations, minibatch=True):
        if minibatch:
            I = self.minibatch_indices
        else:
            I = np.array([x for x in range(self.num_instances_train)])

        p_af = np.copy(self.p_af_train)
        p_raf = np.asarray([self.p_raf_train[self.ratings_train[f],:,f] for f in range(self.num_instances_train)]).T
        probabilities_train = p_af + p_raf
        probabilities_train = self.normalize_safe(probabilities_train, axis=0)
        self.aspects_train_prev = self.aspects_train[:]
        print('probabilities_train', probabilities_train)

        if not self.fix_aspects:
            self.aspects_train[I] = np.array(np.argmax(probabilities_train, axis=0)).flatten()[I]
        else:
            indices_to_update = self.text_reviews_train['value_not_assigned'].copy() & pd.Series([True if x in I else False for x in range(self.num_instances_train)])
            self.aspects_train[indices_to_update] = np.array(np.argmax(probabilities_train, axis=0)).flatten()[indices_to_update]

        frac_match_train = 1.0*sum([(self.aspects_train[i] == self.aspects_train_prev[i]) for i in range(len(self.aspects_train))]) / len(self.aspects_train)
        self.model_assignment_fraction_match_train.append(frac_match_train)

        p_af = np.copy(self.p_af_test)
        p_raf = np.asarray([self.p_raf_test[self.ratings_test[f],:,f] for f in range(self.num_instances_test)]).T
        probabilities_test = p_af + p_raf
        probabilities_test = self.normalize_safe(probabilities_test, axis=0)
        self.aspects_test_prev = self.aspects_test[:]
        print('probabilities_test', probabilities_test)

        if not self.fix_aspects:
            self.aspects_test = np.array(np.argmax(probabilities_test, axis=0)).flatten()
        else:
            indices_to_update = self.text_reviews_test['value_not_assigned'].copy()
            self.aspects_test[indices_to_update] = np.array(np.argmax(probabilities_test, axis=0)).flatten()[indices_to_update]

        frac_match_test = 1.0*sum([(self.aspects_test[i] == self.aspects_test_prev[i]) for i in range(len(self.aspects_test))]) / len(self.aspects_test)
        self.model_assignment_fraction_match_test.append(frac_match_test)

        probabilities_train_aspect_chosen = np.array(np.max(probabilities_train, axis=0)).flatten()
        probabilities_train_percentile_score = np.percentile(probabilities_train_aspect_chosen, self.outlier_percentile)
        self.outlier_train = (probabilities_train_aspect_chosen < probabilities_train_percentile_score)
        self.outlier_train_probabilities = probabilities_train_aspect_chosen

    def save_sparse_csr(self, filename, array):
        np.savez(filename, data = array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)

    def load_sparse_csr(self, filename):
        loader = np.load(filename)
        return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-k", "--samplefraction", dest="samplefraction", type="float", default=1.0, help="Fraction of Datapoints to Sample")
    parser.add_option("-l", "--trainfraction", dest="trainfraction", type="float", default=0.5, help="Fraction of Datapoints to Train on")

    parser.add_option("-m", "--minibatchsize", dest="minibatchsize", type="int", default=1000, help="Minibatch Size")
    parser.add_option("-j", "--numiterations", dest="numiterations", type="int", default=100, help="Number of Iterations")
    parser.add_option("-z", "--learningrate", dest="learningrate", type="float", default=0.01, help="Learning Rate")

    parser.add_option("-x", "--l1rate", dest="l1rate", type="float", default=0.1, help="L1 Regularization Rate")
    parser.add_option("-y", "--l2rate", dest="l2rate", type="float", default=0.1, help="L2 Regularization Rate")
    parser.add_option("-w", "--detrate", dest="detrate", type="float", default=0.1, help="Determinantal Regularization Rate")

    parser.add_option("-s", "--separatepostags", dest="separatepostags", action="store_true", default=True, help="Whether to separate POS tags")
    parser.add_option("-f", "--fixaspects", dest="fixaspects", action="store_true", default=False, help="Fix Aspects to Observed Company Values")
    parser.add_option("-i", "--includenullvalues", dest="includenullvalues", action="store_true", default=True, help="Include Reviews with Unassigned Company Values")
    parser.add_option("-o", "--outlierpercentile", dest="outlierpercentile", type="float", default=10.0, help="Percent of Outlier Reviews to Hold Out of Parameter Learning")
    
    parser.add_option("-r", "--resultdirec", dest="resultdirec", type="string", default='../results/', help="results directory")
    
    (options, args) = parser.parse_args()
    print(options)

    aspect_sentiment_model = AspectSentimentModel(minibatch_size=options.minibatchsize, 
        fix_aspects=options.fixaspects, include_null_values=options.includenullvalues, 
        outlier_percentile=options.outlierpercentile, separate_pos_tags=options.separatepostags, 
        result_direc=options.resultdirec, sample_fraction=options.samplefraction, train_fraction=options.trainfraction,
        l1rate=options.l1rate, l2rate=options.l2rate, detrate=options.detrate, learningrate=options.learningrate)

    aspect_sentiment_model.load_company_data()
    aspect_sentiment_model.prepare_company_data()
    aspect_sentiment_model.initialize_model()

    init_time = int(round(time.time()))
    aspect_sentiment_model.learn_model(max_iterations=options.numiterations)
    elapsed_time = int(round(time.time())) - init_time
    print('elapsed train time: ', elapsed_time)

    aspect_sentiment_model.save_model()
