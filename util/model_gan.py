import util.cluster as cluster
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from numpy.random import shuffle
from util.util import xavier_init
from icecream import ic
ic.disable()

class CPMGAN():
    """build model
    """
    def __init__(self, view_num, trainLen, testLen, layer_size, lsd_dim=128, learning_rate=[0.001, 0.001]):
        """
        :param learning_rate:learning rate of network and h 网络和隐空间数据的学习率
        :param view_num:view number 视图数
        :param layer_size:node of each net 每个网络的节点数
        :param lsd_dim:latent space dimensionality 隐空间数据维度
        :param trainLen:training dataset samples 训练数据集样本数
        :param testLen:testing dataset samples 测试数据集样本数
        """
        # 初始化参数
        self.view_num = view_num
        self.layer_size = layer_size
        self.lsd_dim = lsd_dim
        self.trainLen = trainLen
        self.testLen = testLen

        # 初始化隐空间数据：[batch_size, self.lsd_dim]
        self.h_train, self.h_train_update = self.H_init('train')
        self.h_test, self.h_test_update = self.H_init('test')
        self.h = tf.concat([self.h_train, self.h_test], axis=0)
        self.h_index = tf.placeholder(tf.int32, shape=[None, 1], name='h_index') # 隐空间数据索引
        self.h_temp = tf.gather_nd(self.h, self.h_index) # 从h中根据索引获得隐空间数据

        # 初始化输入数据(按视图初始化)：[batch_size, self.layer_size[v_num][-1]]
        self.input = dict()
        self.sn = dict()
        for v_num in range(self.view_num):
            self.input[str(v_num)] = tf.placeholder(tf.float32, shape=[None, self.layer_size[v_num][-1]],
                                                    name='input' + str(v_num))
            self.sn[str(v_num)] = tf.placeholder(tf.float32, shape=[None, 1], name='sn' + str(v_num))
        # ground truth 真实标签
        self.gt = tf.placeholder(tf.int32, shape=[None], name='gt')

        # Build the CPM-GAN model
        self.train_op, self.loss = self.build_model([self.h_train_update, self.h_test_update], learning_rate)

        # open session
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, h_update, learning_rate):
        # 初始化GAN网络
        G = dict()
        D_x = dict()
        D_G = dict()
        for v_num in range(self.view_num):
            # 使用生成器代替编码网络
            G[str(v_num)] = self.Generator_net(self.h_temp, v_num)
            D_x[str(v_num)] = self.Discriminator_net(self.input[str(v_num)], v_num, reuse=False)
            D_G[str(v_num)] = self.Discriminator_net(G[str(v_num)], v_num, reuse=True)

        # calculate reconstruction loss 计算重建损失
        reco_loss = self.reconstruction_loss(G)
        # calculate adversarial loss 计算对抗损失
        adv_loss = self.adversarial_loss(D_x, D_G)
        
        # Combine reconstruction loss and adversarial loss
        all_loss = reco_loss + adv_loss
        

        # train net operator 训练网络操作
        train_D_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(-all_loss, var_list=tf.get_collection('discriminator'))
        train_G_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(all_loss, var_list=tf.get_collection('generator'))
        # train the latent space data to minimize reconstruction loss and adversarial loss 训练隐空间数据以最小化重建损失和对抗损失
        train_hn_op = tf.train.AdamOptimizer(learning_rate[1]) \
            .minimize(all_loss, var_list=h_update[0])
        # adjust the latent space data
        adj_hn_op = tf.train.AdamOptimizer(learning_rate[0]) \
            .minimize(reco_loss, var_list=h_update[1])
        return [train_D_op, train_G_op, train_hn_op, adj_hn_op], [reco_loss, adv_loss]

    def H_init(self, a):
        # 初始化隐空间数据
        with tf.variable_scope('H' + a):
            # if a == 'train':
            h = tf.Variable(xavier_init(self.trainLen, self.lsd_dim))
            # elif a == 'test':
            #     h = tf.Variable(xavier_init(self.testLen, self.lsd_dim))
            h_update = tf.trainable_variables(scope='H' + a) # 获取可训练变量
        return h, h_update

    # def Encoding_net(self, h, v):
        # In:隐空间数据、视图编号
        # Out:编码网络的输出
        weight = self.initialize_weight(self.layer_size[v], 'encoder')
        layer = tf.matmul(h, weight['w0']) + weight['b0']
        for num in range(1, len(self.layer_size[v])):
            layer = tf.nn.dropout(tf.matmul(layer, weight['w' + str(num)]) + weight['b' + str(num)], 0.9)
        return layer
        
    def Generator_net(self, h, v_num, is_training=True):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            # 该网络的输入是lsd_dim，输出是训练数据各模态的特征数
            # In: 隐空间数据、视图编号
            # Out: 生成器网络的输出
            weight = self.initialize_weight(self.layer_size[v_num], 'generator')
            layer = tf.matmul(h, weight['w0']) + weight['b0']
            for num in range(1, len(self.layer_size[v_num])):
                layer = tf.nn.dropout(tf.matmul(layer, weight['w' + str(num)]) + weight['b' + str(num)], 0.9)
                layer = tf.nn.relu(layer)  # 激活函数
            return layer
    
    def Discriminator_net(self, x, v_num, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            layer_sizes = [self.layer_size[v_num][-1], 128, 64, 1]  # 判别器的网络结构
            weights = self.initialize_D_weight(layer_sizes, scope_name='discriminator')
            layer = x
            for i in range(len(layer_sizes) - 1):
                layer = tf.matmul(layer, weights['w' + str(i)]) + weights['b' + str(i)]
                layer = tf.nn.relu(layer)  # 激活函数
            return tf.nn.sigmoid(layer)
        
    def initialize_D_weight(self, dims_net, scope_name):
        # x->1
        # 初始化网络权重
        # In:网络维度
        # Out:网络权重
        all_weight = dict()
        with tf.variable_scope(scope_name):
            for num in range(1, len(dims_net)):
                all_weight['w' + str(num - 1)] = tf.Variable(xavier_init(dims_net[num - 1], dims_net[num]))
                all_weight['b' + str(num - 1)] = tf.Variable(tf.zeros([dims_net[num]]))
                tf.add_to_collection(scope_name, all_weight['w' + str(num - 1)])
                tf.add_to_collection(scope_name, all_weight['b' + str(num - 1)])
        return all_weight

    def initialize_weight(self, dims_net, scope_name):
        # h->x
        # 初始化网络权重
        # In:网络维度
        # Out:网络权重
        all_weight = dict()
        with tf.variable_scope(scope_name):
            all_weight['w0'] = tf.Variable(xavier_init(self.lsd_dim, dims_net[0]))
            all_weight['b0'] = tf.Variable(tf.zeros([dims_net[0]]))
            tf.add_to_collection(scope_name, all_weight['w' + str(0)])
            tf.add_to_collection(scope_name, all_weight['b' + str(0)])
            for num in range(1, len(dims_net)):
                all_weight['w' + str(num)] = tf.Variable(xavier_init(dims_net[num - 1], dims_net[num]))
                all_weight['b' + str(num)] = tf.Variable(tf.zeros([dims_net[num]]))
                tf.add_to_collection(scope_name, all_weight['w' + str(num)])
                tf.add_to_collection(scope_name, all_weight['b' + str(num)])
        return all_weight

    def adversarial_loss(self, D_x, D_G):
        # $$
        # \begin{aligned}
        # \mathcal{L}_{a d v}= & \sum_{n=1}^N \sum_{v=1}^V \sum_{i=1}^I\left(1-s_{n v}\right)\left[\log D_v\left(\mathbf{x}_i^{(v)} ; \boldsymbol{\Theta}_d^{(v)}\right)\right. \\
        # & \left.+\log \left(1-D_v\left(G_v\left(\mathbf{h}_n ; \boldsymbol{\Theta}_g^{(v)}\right) ; \boldsymbol{\Theta}_d^{(v)}\right)\right)\right],
        # \end{aligned}
        # $$
        adv_loss = 0
        eps = 1e-12  # 避免log(0)的情况
        for v_num in range(self.view_num):
                # 检查D_x和D_G维度是否正确
                ic(D_x[str(v_num)])
                ic(D_G[str(v_num)])
                # 计算对抗损失
                adv_loss += tf.reduce_sum(
                    (1 - self.sn[str(v_num)]) *
                    (tf.log(tf.maximum(D_x[str(v_num)], eps)) +
                        tf.log(tf.maximum(1 - D_G[str(v_num)], eps)))
                )
        # 返回对抗损失，一般是一个负数，因为这里没有取负号
        return adv_loss

    def reconstruction_loss(self, G):
        loss = 0
        for num in range(self.view_num):
            loss += tf.reduce_sum(
                tf.pow(tf.subtract(G[str(num)], self.input[str(num)])
                    , 2.0) * self.sn[str(num)]
            )
        return loss

    def train(self, data, sn, epoch, step=[5, 5]):
        global Reconstruction_LOSS, Adversarial_LOSS
        index = np.array([x for x in range(self.trainLen)])
        shuffle(index)
        sn = sn[index]
        # feed_dict用来传递数据
        feed_dict = {self.input[str(v_num)]: data[str(v_num)][index] for v_num in range(self.view_num)}
        feed_dict.update({self.sn[str(i)]: sn[:, i].reshape(self.trainLen, 1) for i in range(self.view_num)})
        feed_dict.update({self.h_index: index.reshape((self.trainLen, 1))})
        for iter in range(epoch):
            # update the D
            for i in range(step[0]):
                _, Reconstruction_LOSS, Adversarial_LOSS = self.sess.run(
                    [self.train_op[0], self.loss[0], self.loss[1]], feed_dict=feed_dict)
            # update the G
            for i in range(step[1]):
                _, Reconstruction_LOSS, Adversarial_LOSS = self.sess.run(
                    [self.train_op[1], self.loss[0], self.loss[1]], feed_dict=feed_dict)
            # update the hn
            for i in range(step[1]):
                _, Reconstruction_LOSS, Adversarial_LOSS = self.sess.run(
                    [self.train_op[2], self.loss[0], self.loss[1]], feed_dict=feed_dict)
            output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}, Adversarial Loss = {:.4f} " \
                .format((iter + 1), Reconstruction_LOSS, Adversarial_LOSS)
            # output = "Epoch : {:.0f}  ===> All Loss = {:.4f}" \
            #     .format((iter + 1), Reconstruction_LOSS+Adversarial_LOSS)
            
            print(output)

# CPMGAN的测试阶段使用聚类，不需要test方法
    # def test(self, data, sn, gt, epoch):
    #     # 保留CPM-Nets在测试阶段调整hn的操作,分类较弱与CPmnets，可能因为没有监督
    #     feed_dict = {self.input[str(v_num)]: data[str(v_num)] for v_num in range(self.view_num)}
    #     feed_dict.update({self.sn[str(i)]: sn[:, i].reshape(self.testLen, 1) for i in range(self.view_num)})
    #     feed_dict.update({self.gt: gt})
    #     feed_dict.update({self.h_index:
    #                           np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1) + self.trainLen})
    #     for iter in range(epoch):
    #         # update the h
    #         for i in range(5):
    #             _, Reconstruction_LOSS = self.sess.run(
    #                 [self.train_op[3], self.loss[0]], feed_dict=feed_dict)
    #         output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}" \
    #             .format((iter + 1), Reconstruction_LOSS)
    #         print(output)
    
    # def test(self, data, sn, gt, epoch):
    #     # 符合伪代码描述的不进行调整，但是分类性能极差
    #     feed_dict = {self.input[str(v_num)]: data[str(v_num)] for v_num in range(self.view_num)}
    #     feed_dict.update({self.sn[str(i)]: sn[:, i].reshape(self.testLen, 1) for i in range(self.view_num)})
    #     feed_dict.update({self.gt: gt})
    #     feed_dict.update({self.h_index:
    #                           np.array([x for x in range(self.testLen)]).reshape(self.testLen, 1) + self.trainLen})
    #     for iter in range(epoch):
    #         # compute the loss without updating the h
    #         Reconstruction_LOSS, Adversarial_LOSS = self.sess.run(
    #             [self.loss[0], self.loss[1]], feed_dict=feed_dict)
    #         output = "Epoch : {:.0f}  ===> Reconstruction Loss = {:.4f}, Adversarial Loss = {:.4f}" \
    #             .format((iter + 1), Reconstruction_LOSS, Adversarial_LOSS)
    #         print(output)

    def get_h_train(self):
        lsd = self.sess.run(self.h)
        return lsd[0:self.trainLen]

    def get_h_test(self):
        lsd = self.sess.run(self.h)
        return lsd[self.trainLen:]
    