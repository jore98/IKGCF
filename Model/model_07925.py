import logging
import os
import time
from utility.helper import *
import tensorflow as tf
import os
import sys
from utility.helper import *
from utility.batch_test_07925 import *
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Model(object):

    def __init__(self, data_config):

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_kg = data_config['n_kg']
        self.n_relation=data_config['n_relation']
        self.n_fold = 10

        self.norm_adj = data_config['norm_adj']
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.kge_dim = args.kge_embedding

        self.n_layers = data_config['n_layers']

        self.decay = data_config['decay']

        self.self_training=args.self_training

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))

        self.items = tf.placeholder(tf.int32, shape=(None,))
        self.pos_e = tf.placeholder(tf.int32, shape=(None,))
        self.neg_e = tf.placeholder(tf.int32, shape=(None,))



        # initialization of model parameters
        self.weights = self._init_weights()
        
        self.all_embeddings = self._create_rgcf_norm_embed()

        self.ua_embeddings_1, self.ia_embeddings_1, self.ea_embeddings_1 = self.all_embeddings[0]

        self.u_g_embeddings_1 = tf.nn.embedding_lookup(self.ua_embeddings_1, self.users)
        self.pos_i_g_embeddings_1 = tf.nn.embedding_lookup(self.ia_embeddings_1, self.pos_items)
        self.neg_i_g_embeddings_1 = tf.nn.embedding_lookup(self.ia_embeddings_1, self.neg_items)

        self.i_g_embeddings_1 = tf.nn.embedding_lookup(self.ia_embeddings_1, self.items)
        self.pos_e_g_embeddings_1 = tf.nn.embedding_lookup(self.ea_embeddings_1, self.pos_e)
        self.neg_e_g_embeddings_1 = tf.nn.embedding_lookup(self.ea_embeddings_1, self.neg_e)

        self.ua_embeddings_2, self.ia_embeddings_2, self.ea_embeddings_2 = self.all_embeddings[1]

        self.u_g_embeddings_2 = tf.nn.embedding_lookup(self.ua_embeddings_2, self.users)
        self.pos_i_g_embeddings_2 = tf.nn.embedding_lookup(self.ia_embeddings_2, self.pos_items)
        self.neg_i_g_embeddings_2 = tf.nn.embedding_lookup(self.ia_embeddings_2, self.neg_items)

        self.i_g_embeddings_2 = tf.nn.embedding_lookup(self.ia_embeddings_2, self.items)
        self.pos_e_g_embeddings_2 = tf.nn.embedding_lookup(self.ea_embeddings_2, self.pos_e)
        self.neg_e_g_embeddings_2 = tf.nn.embedding_lookup(self.ea_embeddings_2, self.neg_e)


        self.ua_embeddings_3, self.ia_embeddings_3, self.ea_embeddings_3 = self.all_embeddings[2]

        self.u_g_embeddings_3 = tf.nn.embedding_lookup(self.ua_embeddings_3, self.users)
        self.pos_i_g_embeddings_3 = tf.nn.embedding_lookup(self.ia_embeddings_3, self.pos_items)
        self.neg_i_g_embeddings_3 = tf.nn.embedding_lookup(self.ia_embeddings_3, self.neg_items)

        self.i_g_embeddings_3 = tf.nn.embedding_lookup(self.ia_embeddings_3, self.items)
        self.pos_e_g_embeddings_3 = tf.nn.embedding_lookup(self.ea_embeddings_3, self.pos_e)
        self.neg_e_g_embeddings_3 = tf.nn.embedding_lookup(self.ea_embeddings_3, self.neg_e)

        self.ua_embeddings_4, self.ia_embeddings_4, self.ea_embeddings_4 = self.all_embeddings[3]

        self.u_g_embeddings_4 = tf.nn.embedding_lookup(self.ua_embeddings_4, self.users)
        self.pos_i_g_embeddings_4 = tf.nn.embedding_lookup(self.ia_embeddings_4, self.pos_items)
        self.neg_i_g_embeddings_4 = tf.nn.embedding_lookup(self.ia_embeddings_4, self.neg_items)

        self.i_g_embeddings_4 = tf.nn.embedding_lookup(self.ia_embeddings_4, self.items)
        self.pos_e_g_embeddings_4 = tf.nn.embedding_lookup(self.ea_embeddings_4, self.pos_e)
        self.neg_e_g_embeddings_4 = tf.nn.embedding_lookup(self.ea_embeddings_4, self.neg_e)

         #使用第三层和第四层的输出去做预测
        self.batch_ratings_1 = tf.matmul(self.u_g_embeddings_4, self.pos_i_g_embeddings_4, transpose_a=False,
                                          transpose_b=True) + \
                                 tf.matmul(self.u_g_embeddings_3, self.pos_i_g_embeddings_3, transpose_a=False,
                                          transpose_b=True)


        self.loss, self.mf_loss, self.kg_loss, self.emb_loss = self.create_bpr_loss()



        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)


    def _init_weights(self):

        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                    name='user_embedding')
        all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                    name='item_embedding')
        all_weights['e_embedding'] = tf.Variable(initializer([self.n_kg, self.emb_dim]),
                                                 name='e_embedding')
        all_weights['r_embedding'] = tf.Variable(initializer([self.n_relation, self.kge_dim]),
                                                 name='kge_embedding')
        all_weights['trans_W'] = tf.Variable(initializer([self.n_relation, self.emb_dim, self.kge_dim]))
        return all_weights

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items + self.n_kg) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items + self.n_kg
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _create_rgcf_norm_embed(self):

        A_fold_hat = self._split_A_hat(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding'], self.weights['e_embedding']], axis=0)

        all_embeddings = {}

        for k in range(0, self.n_layers):

            temp_embed = []

            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            side_embeddings = tf.concat(temp_embed, 0)

            ego_embeddings = side_embeddings

            u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4 = tf.split(ego_embeddings, [self.n_users, self.n_items, self.n_kg], 0)

            all_embeddings[k] = [u_g_embeddings_4, i_g_embeddings_4, e_g_embeddings_4]

        return all_embeddings

    def create_bpr_loss(self):
         #使用第三层和第四层的输出去做预测
        pos_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.pos_i_g_embeddings_4), axis=1)
        neg_scores_4 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_4, self.neg_i_g_embeddings_4), axis=1)

        pos_scores_kg_4 = tf.reduce_sum(tf.multiply(self.i_g_embeddings_4, self.pos_e_g_embeddings_4), axis=1)
        neg_scores_kg_4 = tf.reduce_sum(tf.multiply(self.i_g_embeddings_4, self.neg_e_g_embeddings_4), axis=1)

        pos_scores_3 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_3, self.pos_i_g_embeddings_3), axis=1)
        neg_scores_3 = tf.reduce_sum(tf.multiply(self.u_g_embeddings_3, self.neg_i_g_embeddings_3), axis=1)

        pos_scores_kg_3 = tf.reduce_sum(tf.multiply(self.i_g_embeddings_3, self.pos_e_g_embeddings_3), axis=1)
        neg_scores_kg_3 = tf.reduce_sum(tf.multiply(self.i_g_embeddings_3, self.neg_e_g_embeddings_3), axis=1)

        regularizer_mf = tf.nn.l2_loss(self.u_g_embeddings_4) + tf.nn.l2_loss(self.pos_i_g_embeddings_4) + \
                      tf.nn.l2_loss(self.neg_i_g_embeddings_4)

        regularizer_kg = tf.nn.l2_loss(self.i_g_embeddings_4) + tf.nn.l2_loss(self.pos_e_g_embeddings_4) + \
                         tf.nn.l2_loss(self.neg_e_g_embeddings_4)

        regularizer = (regularizer_mf + regularizer_kg) / args.batch_size

        mf_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores_4 - neg_scores_4))) + \
                   tf.reduce_mean(tf.nn.softplus(-(pos_scores_3 - neg_scores_3)))
        kg_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores_kg_4 - neg_scores_kg_4))) + \
                   tf.reduce_mean(tf.nn.softplus(-(pos_scores_kg_3 - neg_scores_kg_3)))

        emb_loss = self.decay * regularizer

        loss = mf_loss  + emb_loss +kg_loss

        return loss, mf_loss, kg_loss, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)


if __name__ == '__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    ensureDir('Logs/')

    logfile = 'Logs/' + rq + '.txt'

    fh = logging.FileHandler(logfile, mode='w')

    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    logger.addHandler(fh)

    logger.setLevel(logging.DEBUG)

    logger.info('note: ')

    print('lr-->', args.lr)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for lr in [0.001]:

        n_layers = 4
        print('layer-->', n_layers)
        data_generator.print_statistics()
        config = dict()
        config['n_users'] = data_generator.n_users
        config['n_items'] = data_generator.n_items
        config['n_kg'] = data_generator.n_kg
        config['n_relation'] =data_generator.n_relation
        config['decay'] = 1e-4
        config['n_layers'] = n_layers

        logger.info('#' * 40 + ' dataset={} '.format(args.dataset) + '#' * 40)
        logger.info('#' * 40 + ' n_layers={} '.format(n_layers) + '#' * 40)
        logger.info('#' * 40 + ' decay={} '.format(1e-4) + '#' * 40)
        logger.info('#' * 40 + ' lr={} '.format(lr) + '#' * 40)
        logger.info('-' * 100)
        
        """
        *********************************************************
        Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
        """
        norm_53, norm_54, norm_55 = data_generator.get_adj_mat()

        config['norm_adj'] = norm_54

        print('shape of adjacency', norm_54.shape)

        t0 = time.time()

        model = Model(data_config=config)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.

        """
        *********************************************************
        Train.
        """
        loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
        stopping_step = 0
        should_stop = False
        max_recall, max_precision, max_ndcg, max_hr = 0., 0., 0., 0.
        max_epoch = 0
        self_training=False
        for epoch in range(args.epoch):
            t1 = time.time()
            loss, mf_loss, emb_loss, kg_loss = 0., 0., 0., 0.
            n_batch = data_generator.n_train // args.batch_size + 1

            for idx in range(n_batch):
                users, pos_items,  neg_items = data_generator.sample_u()
                items, pos_e, neg_e= data_generator.sample_i()

                _, batch_loss, batch_mf_loss, batch_kg_loss,batch_emb_loss = sess.run(
                    [model.opt, model.loss, model.mf_loss, model.kg_loss,model.emb_loss],
                    feed_dict={model.users: users,
                               model.pos_items: pos_items,
                               model.neg_items: neg_items,
                               model.items: items,
                               model.pos_e: pos_e,
                               model.neg_e: neg_e})
                loss += batch_loss
                mf_loss += batch_mf_loss
                kg_loss += batch_kg_loss
                emb_loss += batch_emb_loss

            if np.isnan(loss) == True:
                print('ERROR: loss is nan.')
                sys.exit()

            if (epoch + 1) % 20 != 0 or (epoch + 1) < 200:
                if args.verbose > 0 and epoch % args.verbose == 0:
                    perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f +%.5f]' % (
                        epoch, time.time() - t1, loss, mf_loss,kg_loss,emb_loss)
                    print(perf_str)
                continue

            t2 = time.time()
            users_to_test = list(data_generator.test_set.keys())

            ret = test(sess, model, users_to_test)

            t3 = time.time()

            loss_loger.append(loss)
            rec_loger.append(ret['recall'])
            pre_loger.append(ret['precision'])
            ndcg_loger.append(ret['ndcg'])

            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f], precision=[%.5f]' \
                       'ndcg=[%.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss,
                        ret['recall'][0],
                        ret['precision'][0],
                        ret['ndcg'][0])

            print(perf_str)
