import keras
import xlrd
import numpy as np
import jieba
import gensim
import xlwt
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation
from keras.layers import Conv1D,GlobalMaxPooling1D
from keras.layers.advanced_activations import PReLU
from keras.optimizers import rmsprop
from sklearn.utils import check_random_state
from keras.utils import plot_model
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split

data = xlrd.open_workbook('douban.xlsx')
table = data.sheets()[0]
nrows = table.nrows
ncols = table.ncols
mtype = table.col_values(5)
mtype = mtype[1:]
seg_type = []
for i in range(1810):
    seg_type.append(mtype[i].split(sep="\n"))
#电影类型

mreview = table.col_values(10)
mreview = mreview[1:]
for i in range(1810):
    mreview[i] = mreview[i][2:]
seg_list=[]
for i in range(1810):
    seg_list.append(jieba.lcut(mreview[i]))
#电影简介

text_file= open("chinese_stopword.txt","r",encoding="utf-8")
stopword = list(text_file.readlines())
for i in range(len(stopword)):
    stopword[i] = stopword[i].strip("\n")
stopword[-2] = '\n'
stopword[-1] = '\t'
#停顿词

new_review=[]
for re in seg_list:
    new = []
    for word in re:
        if word not in stopword:
            new.append(word)
    new_review.append(new)
#去停顿词

null_number=[]
for i in range(len(new_review)):
    if not new_review[i]:
        null_number.append(i)
for i in null_number:
    del new_review[i]
    del seg_type[i]
#去缺失值

dictionary = corpora.Dictionary(new_review)
corpus = [dictionary.doc2bow(text) for text in new_review]
lda = LdaModel(corpus, num_topics=10,id2word=dictionary, passes=20)
lda.save('movie.lda')
topics_matrix = lda.show_topics(10, 20)
topics_matrix = np.array(topics_matrix)

topic_words = topics_matrix[:,:,1]
for i in topic_words:
    print([str(word) for word in i])
    print()
model = gensim.models.Word2Vec(new_review, min_count=1)
#词向量和LDA

embeddings_index = {}
for mo in new_review:
    for word in mo:
        embeddings_index[word] = model.wv[word]

word_len = map(len, new_review)
word_len_max = max(word_len)
max(new_review, key = len)
wordvec = list()
for mo in new_review:
    embedding_matrix = np.zeros((word_len_max, 100))
    for i,word in enumerate(mo):
        embedding_matrix[i] = embeddings_index[word]
    wordvec.append(embedding_matrix)

type_name = set(item for sublist in seg_type for item in sublist)
type_name1 = list(type_name)
typevec = np.zeros((len(wordvec),len(type_name)))
for i,sen in enumerate(seg_type):
    for word in sen:
        typevec[i][type_name1.index(word)] = 1
np.save('wordvec.npy',wordvec)
np.save('typevec.npy',typevec)
#张量处理

wordvec = np.load('wordvec.npy')
typevec = np.load('typevec.npy')

def separate_sample(x,y):
    ry=[]
    rx=[]
    for y1,y2 in enumerate(y):
        if y2 ==1:
            y3 = np.zeros(32)
            y3[y1]=1
            ry.append(y3)
            rx.append(x)
            break   #只拆一个的话
    return rx,ry
#类别拆分

nwordvec=[]
ntypevec=[]
for x,y in zip(wordvec,typevec):
    rx,ry = separate_sample(x,y)
    nwordvec.extend(rx)
    ntypevec.extend(ry)

ntypevec.tolist()
def to_num(Y):
    ylist = []
    for y in Y:
        ylist.append(y.tolist().index(1))
    return ylist
ntypevec_num = to_num(ntypevec)
#类别数字化

def re_sample(X,y):
    label = np.array(np.unique(y))
    stats_c_ = {}
    maj_n = 0
    X = np.array(X)
    y = np.array(y)
    for i in label:
        nk = sum(y == i)
        stats_c_[i] = nk
        if nk > maj_n:
            maj_n = nk
            maj_c_ = i
    X_resampled = X[y == maj_c_]
    y_resampled = y[y == maj_c_]
    for key in stats_c_.keys():
        if stats_c_[key] != maj_c_:
            num_sample = int(stats_c_[maj_c_] - stats_c_[key])
            random_state = check_random_state(42)
            indx = random_state.randint(low=0, high=(stats_c_[key]-1), size=num_sample)
            X_resampled = np.concatenate((X_resampled,X[y==key]))
            y_resampled = np.concatenate((y_resampled,y[y==key]))
            X_resampled = np.concatenate((X_resampled, X[y == key][indx]))
            y_resampled = np.concatenate((y_resampled, y[y == key][indx]))
    return X_resampled,y_resampled
nwordvec1,ntypevec1_num = re_sample(nwordvec,ntypevec_num)
#类别重抽样

def to_vec(Y):
    ylist = []
    for y in Y:
        y1 = np.zeros(32)
        y1[y] = 1
        ylist.append(y1)
    return np.array(ylist)
ntypevec1 = to_vec(ntypevec1_num)
#类别向量化
np.save('nwordvec1.npy',nwordvec1)
np.save('ntypevec1.npy',ntypevec1)
nwordvec1 = np.load('nwordvec1.npy')
ntypevec1 = np.load('ntypevec1.npy')



model = Sequential()
model.add(Conv1D(input_shape=(134,100),
                 filters = 40,
                 kernel_size = 3,
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(PReLU())
model.add(Dense(32))
model.add(Activation('sigmoid'))
opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='mean_absolute_error',
              optimizer=opt,
              metrics=['mae'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
hist = model.fit(nwordvec1,
                ntypevec1,
                batch_size=32,
                epochs = 1000,
                validation_split=0.7,
                shuffle=True,
                callbacks=[earlyStopping])
print(hist.history)
model.save('cnn.h5')
plot_model(model, to_file='model.png')
#训练CNN  mae=0.0365

model = Sequential()
model.add(Conv1D(input_shape=(134,100),
                 filters = 40,
                 kernel_size = 3,
                 activation='relu',
                 strides=1))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Conv1D(filters = 40,
                 kernel_size = 3,
                 activation='relu',
                 strides=2))
model.add(GlobalMaxPooling1D())
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(PReLU())
model.add(Dense(32))
model.add(Activation('sigmoid'))
opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='mean_absolute_error',
              optimizer=opt,
              metrics=['mae'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')
hist = model.fit(nwordvec1,
                ntypevec1,
                batch_size=32,
                epochs = 1000,
                validation_split=0.7,
                shuffle=True,
                callbacks=[earlyStopping])
print(hist.history)
#2CNN 0.0313

model = Sequential()
model.add(Conv1D(input_shape=(134,100),
                 filters = 100,
                 kernel_size = 3,
                 activation='relu',
                 strides=1))
model.add(Conv1D(filters = 40,
                 kernel_size = 3,
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(BatchNormalization())
model.add(Dense(32))
model.add(Activation('sigmoid'))
opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='mean_absolute_error',
              optimizer=opt,
              metrics=['mae'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
hist = model.fit(nwordvec1,
                ntypevec1,
                batch_size=32,
                epochs = 1000,
                validation_split=0.7,
                shuffle=True,
                callbacks=[earlyStopping])
print(hist.history)
#good CNN 0.0192

X_train, X_test, y_train, y_test = train_test_split(nwordvec1, ntypevec1, test_size=0.33, random_state=42)

#K.image_data_format() == 'channels_first'
model = Sequential()
model.add(Conv1D(input_shape=(134,100),
                 filters=100,
                 kernel_size=3,
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Conv1D(filters=60,
                 kernel_size=3,
                 activation='relu',
                 strides=2))
model.add(MaxPooling1D(3))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(32,activation='softmax'))
opt = rmsprop(lr=0.0001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
hist = model.fit(X_train,
                y_train,
                batch_size=32,
                epochs = 1000,
                validation_split=0.7,
                shuffle=True,
                callbacks=[earlyStopping])
print(hist.history)
score = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#cnn 单个类别

pre = model.predict_proba(wordvec,batch_size=32,verbose=1)

pre1 = []
for i in pre:
    pre1.append(np.argpartition(i, -1)[-5:].tolist())
pre2 = []
for i in pre1:
    pre3 = []
    for j in i:
        pre3.append(type_name1[j])
    pre2.append(pre3)

pre3 = []
for i in pre1:
    pre4 = []
    for j in i:
        pre4.append(type_name1[j])
    pre3.append(pre4)

book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")
for i, l in enumerate(pre3):
    for j, col in enumerate(l):
        sheet1.write(i, j, col)
book.save('type_name.xlsx')