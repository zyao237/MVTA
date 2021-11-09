import numpy as np
import pandas as pd
import math
import sys
import tensorflow as tf
import scipy.sparse as ss
from numpy import linalg as LA
from collections import OrderedDict
from scipy import spatial
import random
import prediction as pred
import pickle

class Learner_NS:
    def __init__(self):
        seed = 1
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(1443)
        self.dir = './'

        self.train_data = Data_prepare(self.dir)
        self.configs = {
                    "att_intra":True,
                    "att_inter":True,
                    "K": 50,  # embedding dim
                    "n_word": self.train_data.n_word,  # number of word
                    "n_rule": self.train_data.n_rule,  # number of conversion rules
                    "n_full_category": self.train_data.n_full_category,
                    "n_targeting_region": self.train_data.n_targeting_region,  # number of targeting audience
                    "n_targeting_device": self.train_data.n_targeting_device,  # number of targeting audience
                    "n_targeting_category": self.train_data.n_targeting_category,  # number of targeting audience
                    "n_campaign_ins": self.train_data.n_campaign_inst,
                    'word_window': 5,
                    'campaign_inst_batch_size': 30,#30
                    'beta': 0.0, # for controlling regularization
                    'loss_type': 'sampled_softmax_loss', #  'nce_loss' or 'sampled_softmax_loss'
                    'word_neg_samples': 200,
                    'rule_neg_samples': 100,
                    'targeting_region_neg_samples': 10,
                    'targeting_device_neg_samples': 2,
                    'targeting_category_neg_samples': 100,
                    'optimizer': 'Adam', #Adam
                    'learning_rate': 0.0005, #0005
                    'sup_learning_rate': 0.00005,
                    'n_steps': 300000, #100000
                    'n_epoch': 50,
                    'initializer_high': 0.01,
                    'initializer_low': -0.01,
                    }
        self.graph_nodes = self._init_graph()
        self.sess = tf.Session(graph=self.graph)

    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # -------------------- embedding placeholder -------------------------
            word_label = tf.placeholder(tf.int32, [None, 1]) # batch size
            word_context = tf.placeholder(tf.int32, [None, None])  # batch size * max length of that batch
            rule_label = tf.placeholder(tf.int32, [None, 1]) # batch size
            rule_context = tf.placeholder(tf.int32, [None, None])  # batch size * max length of that batch
            full_category = tf.placeholder(tf.int32, [None, 1])  # batch size * max length of that batch
            targeting_region_label = tf.placeholder(tf.int32, [None, 1]) # batch size; skip-gram predicted just by campaign instance
            targeting_device_label = tf.placeholder(tf.int32, [None, 1])  # batch size; skip-gram predicted just by campaign instance
            targeting_category_label = tf.placeholder(tf.int32, [None, 1])  # batch size; skip-gram predicted just by campaign instance
            campaign_inst_context = tf.placeholder(tf.int32, [None]) # batch size

            # -------------------- variables -------------------------
            padding_id = 0
            word_mask_array = [[1.]] * padding_id + [[0.]] + [[1.]] * (self.configs['n_word'] - padding_id - 1)
            rule_mask_array = [[1.]] * padding_id + [[0.]] + [[1.]] * (self.configs['n_rule'] - padding_id - 1)
            word_mask_array = np.array(word_mask_array, dtype=np.float32)
            rule_mask_array = np.array(rule_mask_array, dtype=np.float32)

            word_mask_padding_lookup_table = tf.get_variable("word_mask_padding_lookup_table",
                                                        initializer=word_mask_array,
                                                        trainable=False, dtype=tf.float32)
            rule_mask_padding_lookup_table = tf.get_variable("rule_mask_padding_lookup_table",
                                                        initializer=rule_mask_array,
                                                        trainable=False, dtype=tf.float32)

            word_input_emb = tf.get_variable("word_input_emb", dtype=tf.float32,
                                            initializer=tf.random_uniform([self.configs['n_word'], self.configs['K']],
                                                                          self.configs['initializer_low'], self.configs['initializer_high']))
            word_output_weights = tf.get_variable("word_output_emb", dtype=tf.float32,
                                             initializer=tf.random_uniform([self.configs['n_word'], self.configs['K']*2],
                                                                           self.configs['initializer_low'], self.configs['initializer_high']))
            rule_input_emb = tf.get_variable("rule_input_emb", dtype=tf.float32,
                                            initializer=tf.random_uniform([self.configs['n_rule'], self.configs['K']],
                                                                          self.configs['initializer_low'], self.configs['initializer_high']))
            rule_output_weights = tf.get_variable("rule_output_emb", dtype=tf.float32,
                                            initializer=tf.random_uniform([self.configs['n_rule'], self.configs['K']],
                                                                          self.configs['initializer_low'], self.configs['initializer_high']))
            targeting_region_output_weights = tf.get_variable("targeting_region_output_emb", dtype=tf.float32,
                                            initializer=tf.random_uniform([self.configs['n_targeting_region'], self.configs['K']],
                                                                          self.configs['initializer_low'], self.configs['initializer_high']))
            targeting_device_output_weights = tf.get_variable("targeting_device_output_emb", dtype=tf.float32,
                                            initializer=tf.random_uniform([self.configs['n_targeting_device'], self.configs['K']],
                                                                          self.configs['initializer_low'], self.configs['initializer_high']))
            targeting_category_output_weights = tf.get_variable("targeting_category_output_emb", dtype=tf.float32,
                                            initializer=tf.random_uniform([self.configs['n_targeting_category'], self.configs['K']],
                                                                          self.configs['initializer_low'], self.configs['initializer_high']))
            campaign_inst_input_emb = tf.get_variable("campaign_inst_input_emb", dtype=tf.float32,
                                            initializer=tf.random_uniform([self.configs['n_campaign_ins'], self.configs['K']],
                                                                          self.configs['initializer_low'], self.configs['initializer_high']))
            # --------------------------------- inner attention variables ------------------------------
            np_word_att = np.random.normal(0, 0.01, [self.configs['n_word'], self.configs['n_full_category']]).astype(np.float32)
            np_word_att[0][:] = -10.
            np_word_att_bias = np.zeros((self.configs['n_full_category'], )).astype(np.float32)
            intra_att_word = tf.get_variable("intra_att_word", dtype=tf.float32, initializer=np_word_att, trainable=True)
            intra_att_word_bias = tf.get_variable("intra_att_word_bias", dtype=tf.float32, initializer=np_word_att_bias, trainable=True)

            np_rule_att = np.random.normal(0, 0.01, [self.configs['n_rule'], self.configs['n_full_category']]).astype(np.float32)
            np_rule_att[0][:] = -10.
            np_rule_att_bias = np.zeros((self.configs['n_full_category'],)).astype(np.float32)
            intra_att_rule = tf.get_variable("intra_att_rule", dtype=tf.float32, initializer=np_rule_att, trainable=True)
            intra_att_rule_bias = tf.get_variable("intra_att_rule_bias", dtype=tf.float32, initializer=np_rule_att_bias, trainable=True)

            # --------------------------------- inter attention variables ------------------------------
            np_word_inter_att = np.random.normal(0, 0.01, [self.configs['n_word'], 1]).astype(np.float32)
            np_rule_inter_att = np.random.normal(0, 0.01, [self.configs['n_rule'], 1]).astype(np.float32)
            inter_att_word = tf.get_variable("inter_att_word", dtype=tf.float32, initializer=np_word_inter_att, trainable=True)
            inter_att_rule = tf.get_variable("inter_att_rule", dtype=tf.float32, initializer=np_rule_inter_att, trainable=True)

            np_targeting_category_inter_att = np.random.normal(0, 0.01, [self.configs['n_targeting_category'], 1]).astype(np.float32)
            inter_att_targeting_category = tf.get_variable("inter_att_targeting_category", dtype=tf.float32, initializer=np_targeting_category_inter_att, trainable=True)

            # -------------------- flow -------------------------
            word_input = tf.nn.embedding_lookup(word_input_emb, word_context)
            rule_input = tf.nn.embedding_lookup(rule_input_emb, rule_context)
            word_mask_padding_input = tf.nn.embedding_lookup(word_mask_padding_lookup_table, word_context)
            rule_mask_padding_input = tf.nn.embedding_lookup(rule_mask_padding_lookup_table, rule_context)
            campaign_inst_input = tf.nn.embedding_lookup(campaign_inst_input_emb, campaign_inst_context)

            multiplier = tf.stack([tf.constant(1), tf.shape(word_context)[1]], 0)
            full_category_indices_matrix = tf.tile(full_category, multiplier)
            full_idx_matrix = tf.stack([word_context, full_category_indices_matrix], 2)
            word_input_att_param = tf.gather_nd(intra_att_word, full_idx_matrix)
            word_input_att_bias = tf.gather(intra_att_word_bias, full_category_indices_matrix)
            word_input_att_logits = tf.math.add(word_input_att_param, word_input_att_bias)
            word_input_att_weights = tf.expand_dims(tf.nn.softmax(word_input_att_logits, axis=1), 2)

            multiplier = tf.stack([tf.constant(1), tf.shape(rule_context)[1]], 0)
            full_category_indices_matrix = tf.tile(full_category, multiplier)
            full_idx_matrix = tf.stack([rule_context, full_category_indices_matrix], 2)
            rule_input_att_param = tf.gather_nd(intra_att_rule, full_idx_matrix)
            rule_input_att_bias = tf.gather(intra_att_rule_bias, full_category_indices_matrix)
            rule_input_att_logits = tf.math.add(rule_input_att_param, rule_input_att_bias)
            rule_input_att_weights = tf.expand_dims(tf.nn.softmax(rule_input_att_logits, axis=1), 2)

            word_input = tf.multiply(word_input, word_mask_padding_input)
            if self.configs['att_intra']:
                word_input = tf.multiply(word_input, word_input_att_weights)
            word_input_aggr = tf.reduce_sum(word_input, 1)
            word_input_final = tf.concat([campaign_inst_input, word_input_aggr], 1)

            rule_input = tf.multiply(rule_input, rule_mask_padding_input)
            if self.configs['att_intra']:
                rule_input = tf.multiply(rule_input, rule_input_att_weights)
            rule_input_aggr = tf.reduce_sum(rule_input, 1)

            rule_input_mask = tf.reduce_max(rule_input_aggr, 1)
            rule_input_mask = tf.expand_dims(rule_input_mask, 1)
            rule_input_count = tf.cast(tf.count_nonzero(rule_input_mask, 1), tf.float32)
            rule_input_count = tf.expand_dims(rule_input_count, 1)

            campaign_inst_input_mask = tf.reduce_max(campaign_inst_input, 1)
            campaign_inst_input_mask = tf.expand_dims(campaign_inst_input_mask, 1)
            campaign_inst_input_count = tf.cast(tf.count_nonzero(campaign_inst_input_mask, 1), tf.float32)
            campaign_inst_input_count = tf.expand_dims(campaign_inst_input_count, 1)

            rule_input_count_final = tf.add(rule_input_count, campaign_inst_input_count)

            rule_input_final = tf.add(rule_input_aggr, campaign_inst_input)
            rule_input_final = tf.realdiv(rule_input_final, rule_input_count_final)

            # inter-attention
            word_output_att_logits = tf.nn.embedding_lookup(inter_att_word, word_label)
            rule_output_att_logits = tf.nn.embedding_lookup(inter_att_rule, rule_label)
            targeting_category_att_logits = tf.nn.embedding_lookup(inter_att_targeting_category, targeting_category_label)

            output_logits_concat = tf.concat([word_output_att_logits, rule_output_att_logits, targeting_category_att_logits],1)
            output_logits_concat = tf.squeeze(output_logits_concat, axis=2)
            output_att_weights = tf.nn.softmax(output_logits_concat, axis = 1)

            word_bias = tf.Variable(tf.zeros([self.configs['n_word']]),trainable=False, dtype=tf.float32)
            rule_bias = tf.Variable(tf.zeros([self.configs['n_rule']]), trainable=False, dtype=tf.float32)
            targeting_category_bias = tf.Variable(tf.zeros([self.configs['n_targeting_category']]), trainable=False, dtype=tf.float32)

            # Compute the loss, using a sample of the negative labels each time.
            if self.configs['loss_type'] == 'sampled_softmax_loss':
                loss_word = tf.nn.sampled_softmax_loss(weights=word_output_weights, biases=word_bias, inputs=word_input_final,
                                                  labels=word_label, num_sampled=self.configs['word_neg_samples'], num_classes=self.configs['n_word'])
                loss_rule = tf.nn.sampled_softmax_loss(weights=rule_output_weights, biases=rule_bias, inputs=rule_input_final,
                                                  labels=rule_label, num_sampled=self.configs['rule_neg_samples'], num_classes=self.configs['n_rule'])
                loss_targeting_category = tf.nn.sampled_softmax_loss(weights=targeting_category_output_weights, biases=targeting_category_bias, inputs=campaign_inst_input,
                                                  labels=targeting_category_label, num_sampled=self.configs['targeting_category_neg_samples'], num_classes=self.configs['n_targeting_category'])
            elif self.configs['loss_type'] == 'nce_loss':
                loss_word = tf.nn.nce_loss(weights=word_output_weights, biases=word_bias, inputs=word_input_final,
                                                  labels=word_label, num_sampled=self.configs['word_neg_samples'], num_classes=self.configs['n_word'])
                loss_rule = tf.nn.nce_loss(weights=rule_output_weights, biases=rule_bias, inputs=rule_input_final,
                                                  labels=rule_label, num_sampled=self.configs['rule_neg_samples'], num_classes=self.configs['n_rule'])
                loss_targeting_category = tf.nn.nce_loss(weights=targeting_category_output_weights, biases=targeting_category_bias, inputs=campaign_inst_input,
                                                  labels=targeting_category_label, num_sampled=self.configs['targeting_category_neg_samples'], num_classes=self.configs['n_targeting_category'])
            loss_word = tf.expand_dims(loss_word, 1)
            loss_rule = tf.expand_dims(loss_rule, 1)
            loss_targeting_category = tf.expand_dims(loss_targeting_category, 1)

            loss_concat = tf.concat([loss_word, loss_rule, loss_targeting_category],1)
            if self.configs['att_inter']:
                loss_unsupervised = tf.multiply(loss_concat, output_att_weights)
            else:
                loss_unsupervised = loss_concat
            loss_unsupervised = tf.expand_dims(tf.reduce_sum(loss_unsupervised, axis=1),1)
            loss_unsupervised_value = tf.reduce_mean(loss_unsupervised)

            # -------------------- Optimizing -------------------------
            if self.configs['optimizer'] == 'Adagrad':
                train_op_unsup = tf.train.AdagradOptimizer(self.configs['learning_rate']).minimize(loss_unsupervised_value)
            elif self.configs['optimizer'] == 'SGD':
                train_op_unsup = tf.train.GradientDescentOptimizer(self.configs['learning_rate']).minimize(loss_unsupervised_value)
            elif self.configs['optimizer'] == 'Adam':
                train_op_unsup = tf.train.AdamOptimizer(self.configs['learning_rate']).minimize(loss_unsupervised_value)

            # -------------------- emb normalization -------------------------
            norm_campaign_inst = tf.sqrt(tf.reduce_sum(tf.square(campaign_inst_input_emb), 1, keepdims=True))
            normalized_campaign_inst_emb = campaign_inst_input_emb / norm_campaign_inst

            # -------------------- running -------------------------
            # init op
            init_op = tf.global_variables_initializer()
            # create a saver
            saver = tf.train.Saver()
        return {
            'word_label': word_label,
            'word_context': word_context,
            'rule_label': rule_label,
            'rule_context': rule_context,
            'full_category': full_category,
            'targeting_category_label': targeting_category_label,
            'campaign_inst_context': campaign_inst_context,
            'loss_unsupervised': loss_unsupervised_value,
            'loss_word': loss_word,
            'loss_rule': loss_rule,
            'loss_targeting_category': loss_targeting_category,
            'init_op': init_op,
            'train_op_unsup':train_op_unsup,
            'saver': saver,
            'normalized_campaign_inst_emb':normalized_campaign_inst_emb,
            'unnormalized_campaign_inst_emb': campaign_inst_input_emb
        }

    def train_graph(self):
        session = self.sess
        session.run(self.graph_nodes['init_op'])
        average_loss = 0
        average_s_loss = 0
        average_u_loss = 0
        batch_count = 0
        epoch_count = 0 # indicate how many times all campaign instances have been train once
        print("Initialized")
        iter_count = 0
        for step in range(self.configs['n_steps']):
            batch_data, epoch_indicator = self.train_data.generate_batch(self.configs['campaign_inst_batch_size'], self.configs['word_window'])
            feed_dict_unsup = {self.graph_nodes['word_label']: batch_data['word_label_batch'],
                            self.graph_nodes['word_context']: batch_data['word_input_batch'],
                            self.graph_nodes['rule_label']: batch_data['rule_label_batch'],
                            self.graph_nodes['rule_context']: batch_data['rule_input_batch'],
                            self.graph_nodes['full_category']: batch_data['full_category_input_batch'],
                            self.graph_nodes['targeting_category_label']: batch_data['targeting_category_label_batch'],
                            self.graph_nodes['campaign_inst_context']: batch_data['campaign_inst_input_batch']
                         }

            _, u_loss = session.run([self.graph_nodes['train_op_unsup'], self.graph_nodes['loss_unsupervised']], feed_dict=feed_dict_unsup)

            average_u_loss += u_loss
            batch_count += 1
            if epoch_indicator == 1:
                iter_count += 1
                epoch_count += 1
                average_u_loss = average_u_loss / batch_count
                print('Step %d: Overall loss: %f' % (iter_count, average_u_loss))
                average_u_loss = 0
                batch_count = 0

                if epoch_count % self.configs['n_epoch'] == 0:
                    print("start evaluating ...")
                    final_embeddings = session.run(self.graph_nodes['unnormalized_campaign_inst_emb'])
                    test_data = self.train_data.evaluation_campaign_inst_data(final_embeddings)
                    self.evaluate(test_data)
                    # with open('./Evaluation/' + 'vector_campaign_inst_emb_unnorm.pickle', 'wb') as fp:
                    #     pickle.dump(final_embeddings,fp)
                    break

        final_embeddings, ori_embeddings = session.run([self.graph_nodes['normalized_campaign_inst_emb'],
                                                       self.graph_nodes['unnormalized_campaign_inst_emb']])
        print(final_embeddings.shape)
        print("The end")

    def evaluate(self, data_dic):
        evaluator = pred.Evaluator(data_dic)
        print("kNN = 10:")
        evaluator.k_nearest_neighbor(K=10)
        print("kNN = 20:")
        evaluator.k_nearest_neighbor(K=20)
        print("kNN = 30:")
        evaluator.k_nearest_neighbor(K=30)

class Data_prepare:
    def __init__(self, file_dir):
        seed = 1
        random.seed(seed)
        np.random.seed(seed)

        self.dir = file_dir
        self.split_ratio = 0.9

        self.data = pd.read_csv(self.dir + 'campaign_data.csv', skipinitialspace=True)

        self.word_to_id = {}
        self.rule_to_id = {}
        self.full_category_to_id = {}
        self.targeting_region_to_id = {}
        self.targeting_device_to_id = {}
        self.targeting_category_to_id = {}

        self.inst_to_word = {}
        self.inst_to_rule = {}
        self.inst_to_full_category = {}
        self.inst_to_targeting_region = {}
        self.inst_to_targeting_device = {}
        self.inst_to_targeting_category = {}
        self.inst_to_cvr = {}

        # only campaign is not coded
        self.inst_to_campaign = {}
        self.campaign_to_ruleset = {}
        self.campaign_to_instset = {}

        # padding in word and rule
        self.word_to_id['padding'] = 0
        self.rule_to_id['padding'] = 0

        # for training through all data
        # unlike word and rule, the padding inst is the last one
        self.list_campaign_inst = []

        # start reading data
        for idx, row in self.data.iterrows():
            self.list_campaign_inst.append(idx)
            # counting words
            all_sentence = row['ad_content'].replace(";", ":")
            sentence_split = all_sentence.split(':')
            for w in sentence_split:
                # assign id
                if w not in self.word_to_id:
                    self.word_to_id[w] = len(self.word_to_id)
            self.inst_to_word[idx] = [self.word_to_id[i] for i in sentence_split]

            cur_full_category = row['dim_advertiser_complete.category']
            if cur_full_category not in self.full_category_to_id:
                self.full_category_to_id[cur_full_category] = len(self.full_category_to_id)
            self.inst_to_full_category[idx] = self.full_category_to_id[cur_full_category]

            # counting rules
            cur_rule = row['conv_rule_id']
            if cur_rule not in self.rule_to_id:
                self.rule_to_id[cur_rule] = len(self.rule_to_id)
            self.inst_to_rule[idx] = self.rule_to_id[cur_rule]

            # counting targetings
            cur_target_region = row['geo_state']
            if cur_target_region not in self.targeting_region_to_id:
                self.targeting_region_to_id[cur_target_region] = len(self.targeting_region_to_id)
            self.inst_to_targeting_region[idx] = self.targeting_region_to_id[cur_target_region]

            cur_target_device = row['device_type_id']
            if cur_target_device not in self.targeting_device_to_id:
                self.targeting_device_to_id[cur_target_device] = len(self.targeting_device_to_id)
            self.inst_to_targeting_device[idx] = self.targeting_device_to_id[cur_target_device]

            # cur_target_category = row['dim_advertiser_complete.sub_category']
            cur_target_category = row['dim_advertiser_complete.sub_category']+"-"+str(row['device_type_id'])
            if cur_target_category not in self.targeting_category_to_id:
                self.targeting_category_to_id[cur_target_category] = len(self.targeting_category_to_id)
            self.inst_to_targeting_category[idx] = self.targeting_category_to_id[cur_target_category]

            # storing final(smallest) cvr for each ad case
            self.inst_to_cvr[idx] = row['ecvr']

            cur_camp = row['campaign_id']
            self.inst_to_campaign[idx] = cur_camp
            if cur_camp not in self.campaign_to_instset:
                self.campaign_to_instset[cur_camp] =[]
            self.campaign_to_instset[cur_camp].append(idx)
            if cur_camp not in self.campaign_to_ruleset:
                self.campaign_to_ruleset[cur_camp] = set()
            self.campaign_to_ruleset[cur_camp].add(self.rule_to_id[cur_rule])

        self.n_word = len(self.word_to_id)
        self.n_rule = len(self.rule_to_id)
        self.n_full_category = len(self.full_category_to_id)
        self.n_targeting_region = len(self.targeting_region_to_id)
        self.n_targeting_device = len(self.targeting_device_to_id)
        self.n_targeting_category = len(self.targeting_category_to_id)
        self.n_campaign_inst = len(self.list_campaign_inst)
        self.cursor = 0
        self.campaign_inst_shuffle()
        self.epoch_flag = 0 # shows that all campaign inst has been iterated once

        # initialization for supervise learning: CVR predictor
        campaign_list = list(self.campaign_to_instset)
        random.shuffle(campaign_list)
        split_point = int(len(campaign_list) * self.split_ratio)
        test_campaign = set(campaign_list[split_point:])
        print(str(len(test_campaign)) + ' out of ' + str(len(campaign_list)) + ' campaign for test.')
        self.train_campaign_inst = []
        self.test_campaign_inst = []
        for inst in self.list_campaign_inst:
            cur_campaign = self.inst_to_campaign[inst]
            if cur_campaign in test_campaign:
                self.test_campaign_inst.append(inst)
            else:
                self.train_campaign_inst.append(inst)
        print(str(len(self.test_campaign_inst)) + ' out of ' + str(len(self.train_campaign_inst)+len(self.test_campaign_inst)) + ' campaign instance for test.')

    def campaign_inst_shuffle(self):
        random.shuffle(self.list_campaign_inst)
        self.cursor = 0
        # self.seed += 1

    def generate_batch(self, n_sample_inst, word_window):
        if self.cursor + n_sample_inst > self.n_campaign_inst:
            self.campaign_inst_shuffle()
            self.epoch_flag = 1
        indicator = self.epoch_flag
        self.epoch_flag = 0

        campaign_inst_batch = self.list_campaign_inst[self.cursor:self.cursor + n_sample_inst]
        # print(campaign_inst_batch)

        self.cursor += n_sample_inst
        # ----------- embedding input ----------
        word_label_batch = []
        word_input_batch = []
        rule_label_batch = []
        rule_input_batch = []
        full_category_input_batch = []
        targeting_region_label_batch = []
        targeting_device_label_batch = []
        targeting_category_label_batch = []
        campaign_inst_input_batch = []
        # ----------- predictor input ----------
        warm_campaign_inst_mask_batch = []
        cvr_label_batch = []

        for cur_campaign_inst in campaign_inst_batch:
            targeting_region_label = self.inst_to_targeting_region[cur_campaign_inst]
            targeting_device_label = self.inst_to_targeting_device[cur_campaign_inst]
            targeting_category_label = self.inst_to_targeting_category[cur_campaign_inst]

            full_category_input = self.inst_to_full_category[cur_campaign_inst]

            rule_label = self.inst_to_rule[cur_campaign_inst]
            ruleset = self.campaign_to_ruleset[self.inst_to_campaign[cur_campaign_inst]]
            rule_input=[]
            if len(ruleset) > 1:
                rule_input = [x for x in ruleset if x != rule_label]

            words = self.inst_to_word[cur_campaign_inst]

            for word_pos, word in enumerate(words):
                if np.random.rand() < 0.3:
                    continue
                context_pos_start = max(word_pos - word_window, 0)
                context_pos_end = min(word_pos + word_window, len(words)-1)
                word_input = []
                for i in range(context_pos_start, context_pos_end + 1):
                    if i == word_pos:
                        continue
                    word_input.append(words[i])

                # start loading batch input data
                word_label_batch.append(word)
                word_input_batch.append(word_input)
                rule_label_batch.append(rule_label)
                rule_input_batch.append(rule_input.copy())
                full_category_input_batch.append(full_category_input)
                targeting_region_label_batch.append(targeting_region_label)
                targeting_device_label_batch.append(targeting_device_label)
                targeting_category_label_batch.append(targeting_category_label)
                campaign_inst_input_batch.append(cur_campaign_inst)

            if cur_campaign_inst in self.train_campaign_inst:
                warm_campaign_inst_mask_batch.append(cur_campaign_inst)
                cvr_label_batch.append(self.inst_to_cvr[cur_campaign_inst])

        # padding to rules and words
        self.padding(rule_input_batch)
        self.padding(word_input_batch)

        word_label_batch = np.expand_dims(np.array(word_label_batch), axis = 1)
        word_input_batch = np.array(word_input_batch)
        rule_label_batch = np.expand_dims(np.array(rule_label_batch), axis = 1)
        rule_input_batch = np.array(rule_input_batch)
        full_category_input_batch = np.expand_dims(np.array(full_category_input_batch), axis=1)
        targeting_region_label_batch = np.expand_dims(np.array(targeting_region_label_batch), axis=1)
        targeting_device_label_batch = np.expand_dims(np.array(targeting_device_label_batch), axis=1)
        targeting_category_label_batch = np.expand_dims(np.array(targeting_category_label_batch), axis=1)
        campaign_inst_input_batch = np.array(campaign_inst_input_batch)
        warm_campaign_inst_mask_batch = np.array(warm_campaign_inst_mask_batch)
        cvr_label_batch = np.expand_dims(np.array(cvr_label_batch), axis=1)

        return {
            'word_label_batch': word_label_batch,
            'word_input_batch': word_input_batch,
            'rule_label_batch': rule_label_batch,
            'rule_input_batch': rule_input_batch,
            'full_category_input_batch': full_category_input_batch,
            'targeting_region_label_batch': targeting_region_label_batch,
            'targeting_device_label_batch': targeting_device_label_batch,
            'targeting_category_label_batch': targeting_category_label_batch,
            'campaign_inst_input_batch': campaign_inst_input_batch,
            'warm_campaign_inst_mask_batch': warm_campaign_inst_mask_batch,
            'cvr_label_batch': cvr_label_batch
        },indicator

    # input is list of list, the inner lists have different length
    def padding(self, l_of_l):
        max_len = 0
        for i in l_of_l:
            if len(i) > max_len:
                max_len = len(i)
        if max_len == 0:
            for j in l_of_l:
                j.append(0)
        else:
            for i in l_of_l:
                for _ in range(max_len-len(i)):
                    i.append(0)
        return l_of_l

    def evaluation_campaign_inst_data(self, inst_feature_mat):
        x_train = inst_feature_mat[self.train_campaign_inst, :]
        x_test = inst_feature_mat[self.test_campaign_inst, :]
        y_train = [self.inst_to_cvr[inst] for inst in self.train_campaign_inst]
        y_test = [self.inst_to_cvr[inst] for inst in self.test_campaign_inst]

        return {
            'x_test': np.array(x_test),
            'y_test': np.array(y_test),
            'x_train': np.array(x_train),
            'y_train': np.asarray(y_train)
        }

    def generate_multi_hot_feature(self):
        word_mat = np.zeros((self.n_campaign_inst, self.n_word))
        rule_mat = np.zeros((self.n_campaign_inst, self.n_rule))
        targeting_region_mat = np.zeros((self.n_campaign_inst, self.n_targeting_region))
        targeting_device_mat = np.zeros((self.n_campaign_inst, self.n_targeting_device))
        targeting_category_mat = np.zeros((self.n_campaign_inst, self.n_targeting_category))

        for inst in range(self.n_campaign_inst):
            words = self.inst_to_word[inst]
            for w in words:
                word_mat[inst][w] += 1
            rule_mat[inst][self.inst_to_rule[inst]] += 1
            targeting_region_mat[inst][self.inst_to_targeting_region[inst]] += 1
            targeting_device_mat[inst][self.inst_to_targeting_device[inst]] += 1
            targeting_category_mat[inst][self.inst_to_targeting_category[inst]] += 1

        multi_hot_encode = np.concatenate((word_mat, rule_mat, targeting_category_mat), axis=1)
        multi_hot_encode = multi_hot_encode[:,~np.all(multi_hot_encode == 0, axis=0)] # drop zero column made by padding
        cvr_label = np.array([self.inst_to_cvr[inst] for inst in range(self.n_campaign_inst)])

        return {
            'multi_hot_feature': multi_hot_encode,
            'cvr_label': cvr_label,
            'train_inst': self.train_campaign_inst,
            'test_inst': self.test_campaign_inst
        }



if __name__ == '__main__':
    t = Learner_NS()
    t.train_graph()

