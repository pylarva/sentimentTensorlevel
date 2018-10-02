# !/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import jieba # 结巴分词
import socketserver
# gensim用来加载预训练word vector
from gensim.models import KeyedVectors
import warnings

# 使用gensim加载预训练中文分词embedding
cn_model = KeyedVectors.load_word2vec_format('chinese_word_vectors/sgns.zhihu.bigram', binary=False)

# 由此可见每一个词都对应一个长度为300的向量
embedding_dim = cn_model['山东大学'].shape[0]
print('词向量的长度为{}'.format(embedding_dim))
# print(cn_model['山东大学'])

# 计算相似度
print(cn_model.similarity('橘子', '橙子'))

# 找出最相近的词，余弦相似度
print(cn_model.most_similar(positive=['大学'], topn=10))

# 找出不同的词
test_words = '老师 会计师 程序员 律师 医生 老人'
test_words_result = cn_model.doesnt_match(test_words.split())
print('在 %s 中:\n不是同一类别的词为: %s' % (test_words, test_words_result))

pos_txts = os.listdir('pos')
neg_txts = os.listdir('neg')

print("样本总共: %s" % str(len(pos_txts) + len(neg_txts)))

# 现在我们将所有的评价内容放置到一个list里
train_texts_orig = []

# 添加完所有样本之后，train_texts_orig为一个含有4000条文本的list
# 其中前2000条文本为正面评价，后2000条为负面评价
for i in range(len(pos_txts)):
    with open('pos/'+pos_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()
for i in range(len(neg_txts)):
    with open('neg/'+neg_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()

print('总共 %s 条内容' % len(train_texts_orig))

# 使用tensorflow的keras接口来建模
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding, LSTM, Bidirectional
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# 进行分词和tokenize
# train_tokens是一个长长的list，其中含有4000个小list，对应每一条评价
train_tokens = []
for text in train_texts_orig:
    # 去掉标点
    text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
    # 结巴分词
    cut = jieba.cut(text)
    # 结巴分词的输出结果为一个生成器
    # 把生成器转换为list
    cut_list = [ i for i in cut ]
    for i, word in enumerate(cut_list):
        try:
            # 将词转换为索引index
            cut_list[i] = cn_model.vocab[word].index
        except KeyError:
            # 如果词不在字典中，则输出0
            cut_list[i] = 0
    train_tokens.append(cut_list)

# 索引长度标准化
# 获得所有tokens的长度
num_tokens = [len(tokens) for tokens in train_tokens ]
num_tokens = np.array(num_tokens)

print("平均tokens的长度:%s" % np.mean(num_tokens))
print("最长的评价tokens的长度:%s" % np.max(num_tokens))

# plt.hist(np.log(num_tokens), bins=100)
# plt.xlim((0,10))
# plt.ylabel('number of tokens')
# plt.xlabel('length of tokens')
# plt.title('Distribution of tokens length')
# plt.show()

# 取tokens平均值并加上两个tokens的标准差，
# 假设tokens长度的分布为正态分布，则max_tokens这个值可以涵盖95%左右的样本
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
print(max_tokens)

# 取tokens的长度为236时，大约95%的样本被涵盖
# 我们对长度不足的进行padding，超长的进行修剪
np.sum(num_tokens < max_tokens) / len(num_tokens)


# 反向tokenize
# 用来将tokens转换为文本
def reverse_tokens(tokens):
    text = ''
    for i in tokens:
        if i != 0:
            text = text + cn_model.index2word[i]
        else:
            text = text + ' '
    return text


# 经过tokenize再恢复成文本
# 可见标点符号都没有了
reverse = reverse_tokens(train_tokens[0])
print(reverse)


# 原始文本
print(train_texts_orig[0])

# 只使用前20000个词
num_words = 50000
# 初始化embedding_matrix，之后在keras上进行应用
embedding_matrix = np.zeros((num_words, embedding_dim))
# embedding_matrix为一个 [num_words，embedding_dim] 的矩阵
# 维度为 50000 * 300
for i in range(num_words):
    embedding_matrix[i,:] = cn_model[cn_model.index2word[i]]
embedding_matrix = embedding_matrix.astype('float32')

# 检查index是否对应，
# 输出300意义为长度为300的embedding向量一一对应
np.sum(cn_model[cn_model.index2word[333]] == embedding_matrix[333])


# embedding_matrix的维度，
# 这个维度为keras的要求，后续会在模型中用到
print(embedding_matrix.shape)


# 进行padding和truncating， 输入的train_tokens是一个list
# 返回的train_pad是一个numpy array
train_pad = pad_sequences(train_tokens, maxlen=max_tokens, padding='pre', truncating='pre')

# 超出五万个词向量的词用0代替
train_pad[train_pad >= num_words] = 0

# 可见padding之后前面的tokens全变成0，文本在最后面
# train_pad[33]

# 准备target向量，前2000样本为1，后2000为0
train_target = np.concatenate((np.ones(2000), np.zeros(2000)))

# 进行训练和测试样本的分割
from sklearn.model_selection import train_test_split

# 90%的样本用来训练，剩余10%用来测试
X_train, X_test, y_train, y_test = train_test_split(train_pad,
                                                    train_target,
                                                    test_size=0.1,
                                                    random_state=12)

# 查看训练样本，确认无误
print(reverse_tokens(X_train[35]))
print('class: ', y_train[35])

# 用LSTM对样本进行分类
model = Sequential()

# 模型第一层为embedding
model.add(Embedding(num_words,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_tokens,
                    trainable=False))

model.add(Bidirectional(LSTM(units=32, return_sequences=True)))
model.add(LSTM(units=16, return_sequences=False))

model.add(Dense(1, activation='sigmoid'))
# 我们使用adam以0.001的learning rate进行优化
optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

print(model.summary())

# 建立一个权重的存储点
path_checkpoint = 'sentiment_checkpoint.keras'
checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_loss',
                             verbose=1, save_weights_only=True,
                             save_best_only=True)


# 尝试加载已训练模型
try:
    model.load_weights(path_checkpoint)
    graph = tf.get_default_graph()
except Exception as e:
    graph = tf.get_default_graph()
    print(e)

# 定义early stoping如果3个epoch内validation loss没有改善则停止训练
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

# 自动降低learning rate
lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)

# 定义callback函数
callbacks = [
    earlystopping,
    checkpoint,
    lr_reduction
]

# 开始训练
print('开始训练...')
# model.fit(X_train, y_train,
#           validation_split=0.1,
#           epochs=20,
#           batch_size=128,
#           callbacks=callbacks)

result = model.evaluate(X_test, y_test)
print('Accuracy:{0:.2%}'.format(result[1]))


# def predict_sentiment(text):
#     print(text)
#     # 去标点
#     text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "",text)
#     # 分词
#     cut = jieba.cut(text)
#     cut_list = [i for i in cut]
#     # tokenize
#     for i, word in enumerate(cut_list):
#         try:
#             cut_list[i] = cn_model.vocab[word].index
#         except KeyError:
#             cut_list[i] = 0
#     # padding
#     tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
#                            padding='pre', truncating='pre')
#     # 预测
#     result = model.predict(x=tokens_pad)
#     coef = result[0][0]
#     if coef >= 0.5:
#         sed = '是一例正面评价, output=%.2f' % coef
#     else:
#         sed = '是一例负面评价, output=%.2f' % coef
#     print(sed)
#     return sed
#
# test_list = [
#     '你好啊',
#     '酒店设施不是新的，服务态度很不好',
#     '酒店卫生条件非常不好',
#     '床铺非常舒适',
#     '房间很凉，不给开暖气',
#     '房间很凉爽，空调冷气很足',
#     '酒店环境不好，住宿体验很不好',
#     '房间隔音不到位' ,
#     '晚上回来发现没有打扫卫生',
#     '因为过节所以要我临时加钱，比团购的价格贵'
# ]
# for text in test_list:
#     predict_sentiment(text)


class MyServer(socketserver.BaseRequestHandler):
    """
    socketserver 服务端
    """
    def handle(self):
        conn = self.request
        conn.sendall('欢迎访问socketserver服务器！'.encode())
        while True:
            data = conn.recv(1024).decode()
            if data == "exit":
                print("断开与%s的连接！" % (self.client_address,))
                break
            ret1 = self.predict_sentiment("晚上回来发现没有打扫卫生")
            print(ret1)
            ret = self.predict_sentiment(data)
            print("%s" % data)
            print("%s" % ret)
            conn.sendall(('%s' % ret).encode())


    def predict_sentiment(self, text):
        print(text)
        # 去标点
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+", "", text)
        # 分词
        cut = jieba.cut(text)
        cut_list = [i for i in cut]
        # tokenize
        for i, word in enumerate(cut_list):
            try:
                cut_list[i] = cn_model.vocab[word].index
            except KeyError:
                cut_list[i] = 0
        # padding
        tokens_pad = pad_sequences([cut_list], maxlen=max_tokens,
                                   padding='pre', truncating='pre')
        # 预测
        with graph.as_default():
            result = model.predict(x=tokens_pad)
        coef = result[0][0]
        if coef >= 0.5:
            sed = '是一例正面评价, output=%.2f' % coef
        else:
            sed = '是一例负面评价, output=%.2f' % coef
        print(sed)
        return sed


server = socketserver.ThreadingTCPServer(('127.0.0.1', 9997), MyServer)
server.serve_forever()














