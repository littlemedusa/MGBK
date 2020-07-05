import json
import time
import numpy as np
import math
from sklearn.model_selection import train_test_split
import scipy.sparse as sp 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import NMF
import tensorflow as tf #
from sklearn.utils import shuffle
from datetime import datetime
import os
from nltk.corpus import stopwords
import jieba
from gensim import corpora,models
import numpy as np
import json
from bayes_opt import BayesianOptimization

#基本设定
def settings():
    # 取出来一批小白鼠，n是取出的条目数量 
    n = 10000
    m = n/100
    #确定隐变量的个数
    K = 10
    learning_rate = 0.00005
    batch_size = 32
    epoches = 500
    hidden_size = 64
    last_hidden_size = 32
    train_ratio = 0.7
    merge = "attention"
    
    return n, m, K, learning_rate, batch_size, epoches, hidden_size,\
    last_hidden_size, train_ratio, merge

#读入数据
def read_yelp(string):
    file_yelp_string = "C:\\dataset\\yelp-dataset\\yelp_academic_dataset_%s.json "%string
    data_yelp_string = []
    #读n条string
    start_time = time.time()
    with open(file_yelp_string,'r',encoding='gb18030',errors='ignore') as f:
        # 一行一行读
        for j in range(n):
            # 将josn字符串转化为dict字典
            line = f.readline()
            if(line):
                k = json.loads(line)
                data_yelp_string.append(k)
        f.close()
    #print(data_yelp_string[0])    
    #print(len(data_yelp_string))
    end_time = time.time()
    #print("Time: %f s" %(end_time - start_time))
    return data_yelp_string

#收集review信息，把user和item重编号（不能一口气读完user和Business数据集的弊端）
def collect_yelp(string):
    yelp_string_id = set()
    for j in range(n):
        yelp_string_id.add(data_yelp_review[j]['%s_id'%string])
    yelp_string_id = list(yelp_string_id)
    #print(yelp_string_id[0:9])
    return yelp_string_id

#将user和item的长度统一，空余部分用-1补齐，为DNN训练做准备
def samelength(A,B):
    if(len(A)<len(B)):
        for j in range(len(B)-len(A)):
            A.append(-1)
    else:
        for j in range(len(A)-len(B)):
            B.append(-1)

    return A, B

#把训练集和测试集的user,item信息转化为编号信息
def numeralization_yelp():
    print("Numeralization")
    start_time = time.time()
    yelp_train_user_id = [] 
    yelp_train_item_id = [] 
    yelp_test_user_id = [] 
    yelp_test_item_id = []
    yelp_train_star = [] 
    yelp_test_star = []

    #训练集
    for j in range(len(data_yelp_review_train)):
        #提取数据
        user = data_yelp_review_train[j]['user_id']
        item = data_yelp_review_train[j]['business_id']
        star = data_yelp_review_train[j]['stars']
        yelp_train_user_id.append(yelp_user_id.index(user))
        yelp_train_item_id.append(yelp_item_id.index(item))
        yelp_train_star.append(star)
        #计时器
        if(j%(10*m) == 0):
            print(j/m,"%")

    #测试集(数据集补完计划)
    for j in range(len(data_yelp_review_test)):
        #提取数据
        user = data_yelp_review_test[j]['user_id']
        item = data_yelp_review_test[j]['business_id']
        star = data_yelp_review_test[j]['stars']
        yelp_test_user_id.append(yelp_user_id.index(user))
        yelp_test_item_id.append(yelp_item_id.index(item))
        yelp_test_star.append(star)
        #计时器
        if(j%(10*m) == 0):
            print(j/m+70,"%")
    end_time = time.time()
    print("Time: %f s"%(end_time - start_time))
    return yelp_train_user_id, yelp_train_item_id,\
    yelp_test_user_id, yelp_test_item_id, yelp_train_star, yelp_test_star

#拼装M1
def assemble_yelp_M1():
    #M1
    print("Assemble M1")
    start_time = time.time()
    u = len(yelp_user_id)
    i = len(yelp_item_id)
    # M1_test = sp.csr_matrix((stars_test,(user_test_id,item_test_id)),shape = (u,i))
    # M1_test_onehot = sp.csr_matrix((onehot,(user_test_id,item_test_id)),shape = (u,i))
    yelp_M1 = sp.csr_matrix((yelp_train_star,(yelp_train_user_id,yelp_train_item_id)),shape = (u,i))
    yelp_M1_norating = sp.csr_matrix(([1]*len(yelp_train_star),(yelp_train_user_id,yelp_train_item_id)),shape = (u,i))
    end_time = time.time()
    print("Time: %f s" %(end_time - start_time))
    return yelp_M1, yelp_M1_norating

#拼装M2 #好慢啊
def assemble_yelp_M2():
    #M2
    print("Assemble M2")
    start_time = time.time()
    u = len(yelp_user_id)
    i = len(yelp_item_id)
    user_row = []
    user_col = []

    #这乱七八糟的写的什么玩意
    for j in range(n):
        user = data_yelp_user[j]['user_id']
        #数据不全，需要判断
        if(user in yelp_user_id):
            user_id = yelp_user_id.index(user)
            user_friend = data_yelp_user[j]['friends']
            user_friend = user_friend.split(",")
            for k in range(len(user_friend)):
                if(user_friend[k] in yelp_user_id):
                    user_friend_id = yelp_user_id.index(user_friend[k])
                    user_row.append(user_id)
                    user_col.append(user_friend_id)
        if(j%(10*m)==0):
            print(j/m,"%")

    uu = sp.csr_matrix(([1]*len(user_row),(user_row,user_col)),shape = (u,u))
    uid = sp.csr_matrix(([1]*u,(np.arange(u),np.arange(u))),shape = (u,u)) #是u_identity的缩写,才不是B站账号呢
    uu = uu - uid
    ui_total = uu.dot(d['yelp_M1'])
    yelp_M2_norating = uu.dot(d['yelp_M1_norating'])
    row, column, star_total = sp.find(ui_total)
    row, column, star_multi = sp.find(yelp_M2_norating)
    star_average = [a/b for a,b in zip(star_total, star_multi)]
    #print(star_average)
    yelp_M2 = sp.csr_matrix((star_average,(row,column)),shape = (u,i))
    end_time = time.time()
    print("Time: %f s" %(end_time - start_time))
    return yelp_M2, yelp_M2_norating

#拼装M3
def assemble_yelp_M3():
    #M3
    print("Assemble M3")
    start_time = time.time()
    u = len(yelp_user_id)
    i = len(yelp_item_id)

    # M3与M1不刻画重复信息
    uu = d['yelp_M1_norating'].dot(d['yelp_M1_norating'].T)
    uid = sp.csr_matrix(([1]*u,(np.arange(u),np.arange(u))),shape = (u,u)) #是u_identity的缩写,才不是B站账号呢
    uu = uu - uid

    ui_total = uu.dot(d['yelp_M1'])
    yelp_M3_norating = uu.dot(d['yelp_M1_norating'])
    row, column, star_total = sp.find(ui_total)
    row, column, star_multi = sp.find(yelp_M3_norating)
    star_average = [a/b for a,b in zip(star_total, star_multi)]
    yelp_M3 = sp.csr_matrix((star_average,(row,column)),shape = (u,i))
    end_time = time.time()
    print("Time: %f s" %(end_time - start_time))
    return yelp_M3, yelp_M3_norating

#拼装M4
def assemble_yelp_M4():
    #M4
    print("Assemble M4")
    start_time = time.time()
    u = len(yelp_user_id)
    i = len(yelp_item_id)
    cate_set = set()
    cate_sum = [0]*i
    for j in range(n):
        cate = data_yelp_business[j]['categories']
        if(cate != None):
            cate = cate.split(",")
            for k in range(len(cate)):
                cate_set.add(cate[k])

    cate_list = list(cate_set)
    c = len(cate_list)
    belongto_row = []
    belongto_column = []
    for j in range(n):
        if(data_yelp_business[j]['business_id'] in yelp_item_id):
            cate = data_yelp_business[j]['categories']
            if(cate != None):
                cate = cate.split(",")
                item_id = yelp_item_id.index(data_yelp_business[j]['business_id'])
                cate_sum[item_id] = len(cate)
                for k in range(len(cate)):
                    cate_id = cate_list.index(cate[k])
                    belongto_row.append(item_id)
                    belongto_column.append(cate_id)

    ic = sp.csr_matrix(([1]*len(belongto_row),(belongto_row,belongto_column)),shape = (i,c))
    ii_old = ic.dot(ic.T)
    row, column, _sum = sp.find(ii_old)
    for j in range(len(_sum)):
        _id = row[j]
        _sum[j] = _sum[j] / cate_sum[_id]

    #重构ii，0-1标准化,并去掉M1信息
    ii = sp.csr_matrix((_sum,(row,column)),shape = (i,i))
    iid = sp.csr_matrix(([1]*i,(np.arange(i),np.arange(i))),shape = (i,i))
    ii = ii - iid

    ui_total = d['yelp_M1'].dot(ii)
    yelp_M4_norating = d['yelp_M1_norating'].dot(ii)
    row, column, star_total = sp.find(ui_total)
    row, column, star_multi = sp.find(yelp_M4_norating)
    star_average = [a/b for a,b in zip(star_total, star_multi)]
    yelp_M4 = sp.csr_matrix((star_average,(row,column)),shape = (u,i))
    end_time = time.time()
    print("Time: %f s" %(end_time - start_time))
    return yelp_M4, yelp_M4_norating

#拼装M5
def assemble_yelp_M5():
    #M5
    print("Assemble M5")
    start_time = time.time()
    u = len(yelp_user_id)
    i = len(yelp_item_id)
    city_set = set()
    for j in range(n):
        city = data_yelp_business[j]['city']
        city_set.add(city)

    city_list = list(city_set)
    c = len(city_list)
    locatein_row = []
    locatein_column = []
    for j in range(n):
        if(data_yelp_business[j]['business_id'] in yelp_item_id):
            city = data_yelp_business[j]['city']
            city_id = city_list.index(city)
            item_id = yelp_item_id.index(data_yelp_business[j]['business_id'])
            locatein_row.append(item_id)
            locatein_column.append(city_id)

    ic = sp.csr_matrix(([1]*len(locatein_row),(locatein_row,locatein_column)),shape = (i,c))
    ii = ic.dot(ic.T)
    iid = sp.csr_matrix(([1]*i,(np.arange(i),np.arange(i))),shape = (i,i))
    ii = ii - iid

    ui_total = d['yelp_M1'].dot(ii)
    yelp_M5_norating = d['yelp_M1_norating'].dot(ii)
    row, column, star_total = sp.find(ui_total)
    row, column, star_multi = sp.find(yelp_M5_norating)
    star_average = [a/b for a,b in zip(star_total, star_multi)]
    yelp_M5 = sp.csr_matrix((star_average,(row,column)),shape = (u,i))
    end_time = time.time()
    print("Time: %f s" %(end_time - start_time))
    return yelp_M5, yelp_M5_norating

#拼装M6
def assemble_yelp_M6():
    #M6
    print("Assemble M6")
    start_time = time.time()
    u = len(yelp_user_id)
    i = len(yelp_item_id)
    state_set = set()
    for j in range(n):
        state = data_yelp_business[j]['state']
        state_set.add(state)

    state_list = list(state_set)
    s = len(state_list)
    locatein_row = []
    locatein_column = []
    for j in range(n):
        if(data_yelp_business[j]['business_id'] in yelp_item_id):
            state = data_yelp_business[j]['state']
            state_id = state_list.index(state)
            item_id = yelp_item_id.index(data_yelp_business[j]['business_id'])
            locatein_row.append(item_id)
            locatein_column.append(state_id)

    _is = sp.csr_matrix(([1]*len(locatein_row),(locatein_row,locatein_column)),shape = (i,s))
    ii = _is.dot(_is.T)
    iid = sp.csr_matrix(([1]*i,(np.arange(i),np.arange(i))),shape = (i,i))
    ii = ii - iid

    ui_total = d['yelp_M1'].dot(ii)
    yelp_M6_norating = d['yelp_M1_norating'].dot(ii)
    row, column, star_total = sp.find(ui_total)
    row, column, star_multi = sp.find(yelp_M6_norating)
    star_average = [a/b for a,b in zip(star_total, star_multi)]
    yelp_M6 = sp.csr_matrix((star_average,(row,column)),shape = (u,i))
    end_time = time.time()
    print("Time: %f s" %(end_time - start_time))
    return yelp_M6, yelp_M6_norating        

#拼装M7
def assemble_yelp_M7():
    #M7
    print("Assemble M7")
    start_time = time.time()
    u = len(yelp_user_id)
    i = len(yelp_item_id)
    stars_set = set()
    for j in range(n):
        stars = data_yelp_business[j]['stars']
        stars_set.add(stars)

    stars_list = list(stars_set)
    s = len(stars_list)
    obtain_row = []
    obtain_column = []
    for j in range(n):
        if(data_yelp_business[j]['business_id'] in yelp_item_id):
            stars = data_yelp_business[j]['stars']
            stars_id = stars_list.index(stars)
            item_id = yelp_item_id.index(data_yelp_business[j]['business_id'])
            obtain_row.append(item_id)
            obtain_column.append(stars_id)

    _is = sp.csr_matrix(([1]*len(obtain_row),(obtain_row,obtain_column)),shape = (i,s))
    ii = _is.dot(_is.T)
    iid = sp.csr_matrix(([1]*i,(np.arange(i),np.arange(i))),shape = (i,i))
    ii = ii - iid

    ui_total = d['yelp_M1'].dot(ii)
    yelp_M7_norating = d['yelp_M1_norating'].dot(ii)
    row, column, star_total = sp.find(ui_total)
    row, column, star_multi = sp.find(yelp_M7_norating)
    star_average = [a/b for a,b in zip(star_total, star_multi)]
    yelp_M7 = sp.csr_matrix((star_average,(row,column)),shape = (u,i))
    end_time = time.time()
    print("Time: %f s" %(end_time - start_time))
    return yelp_M7, yelp_M7_norating

#拼装M8
def assemble_yelp_M8():
    print("Assemble M8")
    start_time = time.time()
    num_topic = 10
    rwtext = []
    u = len(yelp_user_id)
    i = len(yelp_item_id)
    l = len(yelp_train_number)
    for j in range(l):
        _j = yelp_train_number[j]
        rwtext.append(data_yelp_review[_j]['text'])
    ra = np.zeros([l,num_topic])
    rb = [0]*l
    ldaOut,corpus_lda = grouptopic(rwtext,num_topic)
    #这个ra不太对啊，有点问题
    for j in range(l):
        _j = yelp_train_number[j]
        for k in range(len(corpus_lda[j])):
            ra[j][corpus_lda[j][k][0]] = math.fabs(float("{0:.2f}".format(corpus_lda[j][k][1])))
        rb[j] = np.argmax(ra[j])

    row= []
    column = []
    for j in range(l):
        _j = yelp_train_number[j]
        user = data_yelp_review[_j]['user_id']
        user_id = yelp_user_id.index(user)
        topic = rb[j]
        row.append(user_id)
        column.append(topic)
    ua = sp.csr_matrix((rb,(row,column)),shape = (u,num_topic))

    uu = ua.dot(ua.T)
    uid = sp.csr_matrix(([1]*u,(np.arange(u),np.arange(u))),shape = (u,u))
    uu = uu - uid
    ui_total = uu.dot(d['yelp_M1'])
    yelp_M8_norating = uu.dot(d['yelp_M1_norating'])
    row, column, star_total = sp.find(ui_total)
    row, column, star_multi = sp.find(yelp_M8_norating)
    star_average = [a/b for a,b in zip(star_total, star_multi)]
    yelp_M8 = sp.csr_matrix((star_average,(row,column)),shape = (u,i))
    end_time = time.time()
    print("Time: %f s" %(end_time - start_time))
    return yelp_M8, yelp_M8_norating

#拼装M9
def assemble_yelp_M9():
    print("Assemble M9")
    start_time = time.time()
    num_topic = 10
    rwtext = []
    u = len(yelp_user_id)
    i = len(yelp_item_id)
    l = len(yelp_train_number)
    for j in range(l):
        _j = yelp_train_number[j]
        rwtext.append(data_yelp_review[_j]['text'])
    ra = np.zeros([l,num_topic])
    rb = [0]*l
    ldaOut,corpus_lda = grouptopic(rwtext,num_topic)
    #这个ra不太对啊，有点问题
    for j in range(l):
        for k in range(len(corpus_lda[j])):
            ra[j][corpus_lda[j][k][0]] = math.fabs(float("{0:.2f}".format(corpus_lda[j][k][1])))
        rb[j] = np.argmax(ra[j])
        
    row= []
    column = []
    for j in range(l):
        _j = yelp_train_number[j]
        user = data_yelp_review[_j]['user_id']
        user_id = yelp_user_id.index(user)
        topic = rb[j]
        row.append(user_id)
        column.append(topic)
    ua = sp.csr_matrix((rb,(row,column)),shape = (u,num_topic))

    uu_1 = np.dot(ua,ua.T)
    uid = sp.csr_matrix(([1]*u,(np.arange(u),np.arange(u))),shape = (u,u))
    uu_1 = uu_1 - uid
    uu_2 = d['yelp_M1_norating'].dot(d['yelp_M1_norating'].T)
    uu_2 = uu_2 - uid
    uu = uu_1 * uu_2
    ui_total = uu.dot(d['yelp_M1'])
    yelp_M9_norating = uu.dot(d['yelp_M1_norating'])
    row, column, star_total = sp.find(ui_total)
    row, column, star_multi = sp.find(yelp_M9_norating)
    star_average = [a/b for a,b in zip(star_total, star_multi)]
    yelp_M9 = sp.csr_matrix((star_average,(row,column)),shape = (u,i))
    end_time = time.time()
    print("Time: %f s" %(end_time - start_time))
    return yelp_M9, yelp_M9_norating



def dealword(words):
    texts_lower = [[w.lower() for w in word] for word in words]
    english_stopwords = stopwords.words('english')
    texts_filtered = [[word for word in line if not word in english_stopwords] for line in texts_lower]
    english_punctuations = ['/',',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%',' ','-','"','via','across','1','ℓ','\n']
    clean_words = [[word for word in line if not word in english_punctuations] for line in texts_filtered]
    return clean_words

def grouptopic(sentences,num_topic):
    words = []
    for sentence in sentences:
        words.append(list(jieba.cut(sentence)))
    deal_words = dealword(words)
    dic = corpora.Dictionary(deal_words)
    corpus = [dic.doc2bow(text) for text in words]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word = dic, num_topics = num_topic)
    ldaOut = lsi.print_topics(num_topic)
    corpus_lda = lsi[corpus_tfidf]
    return ldaOut,corpus_lda


#慢，但是不占内存
def lfm_sparse(A):
    
    #参数A：表示需要分解的评价矩阵
    #参数K：分解的属性（隐变量）个数
    
    print("M Factorization")
    start_time = time.time()
    matrix_m = A.todense().shape[0]
    matrix_n = A.todense().shape[1]
    row, column, star = sp.find(A)
    alpha = 0.01
    lambda_u = 0.01
    lambda_i = 0.01
    user = np.random.rand(matrix_m,K)
    item = np.random.randn(K,matrix_n)
    user, item = optimizing(step=1000, row=row, column=column, star=star, user=user, item=item, alpha=0.01, beta=0.01, lannbda=0.01)
    #user, item = optimizing(step=1000, row=row, column=column, star=star, user=user, item=item, alpha=0.0001, beta=000.01, lannbda=0.01)
    end_time = time.time()
    print("Time: %f s"%(end_time - start_time))
    return user,item

#梯度下降
def optimizing(step, row, column, star, user, item, alpha, beta, lannbda):
    result = []
    former_loss = 9223372036854775807.0
    ministep = step/100
    warning_line = m
    u = np.sqrt(np.sum(user*user))
    i = np.sqrt(np.sum(item*item))
    for t in range(step):
        loss = lannbda*(u+i)
        ERR = 0
        for i in range(len(row)):
            x = row[i]
            y = column[i]
            err = star[i] -np.dot(user[x],item[:,y])
            ERR += err * err
            for j in range(K):
                g_u = err * item[j][y] - alpha * user[x][j]
                g_i = err * user[x][j] - alpha * item[j][y]
                user[x][j] += min(beta * g_u , warning_line)
                item[j][y] += min(beta * g_i , warning_line)
        loss += np.sqrt(ERR)
        result.append(loss)
        if(t%ministep == 0):
            print(t/ministep, "%") 
        if(loss == 0):
            break 
        elif(math.fabs(former_loss/(loss) - 1) < 1e-4):
            print("Converge!")
            break 
        former_loss = loss
    
    #画图  
    # plt.plot(range(len(result)//2,len(result)),result[len(result)//2:])
    # plt.xlabel("time")
    # plt.ylabel("loss")
    # plt.show()   
    # plt.close()
    
    return user, item


def LFM(D, k, iter_times=1000, alpha=0.01,lannbda=0.01, learn_rate=0.01):
    '''
    此函数实现的是最简单的 LFM 功能
    :param D: 表示需要分解的评价矩阵, type = np.ndarray
    :param k: 分解的隐变量个数
    :param iter_times: 迭代次数
    :param alpha: 正则系数
    :param learn_rate: 学习速率
    '''
    print("M Factorization")
    start_time = time.time()
    assert type(D) == np.ndarray
    m, n = D.shape  # D size = m * n
    M = np.float32(D>0)
    user = np.random.rand(m, k)    
    item = np.random.randn(k, n)
    former_loss = 9223372036854775807.0
    u = np.sqrt(np.sum(user * user))
    i = np.sqrt(np.sum(item * item))
    for t in range(iter_times):
        D_est = np.matmul(user, item)
        ERR = M * (D - D_est)
        user_grad = -2 * np.matmul(ERR, item.transpose()) + 2 * alpha * user
        item_grad = -2 * np.matmul(user.transpose(), ERR) + 2 * alpha * item
        user = user - learn_rate * user_grad
        item = item - learn_rate * item_grad

        loss = np.sqrt(np.sum(ERR*ERR)) + lannbda*u + lannbda*i

        if(t%(iter_times/100) == 0):
            print(t/(iter_times/100), "%") 
        if(loss == 0):
            break 
        elif(math.fabs(former_loss/(loss) - 1) < 1e-3):
            print("Converge!")
            break 
        former_loss = loss

    end_time = time.time()
    print("Time: %f s"%(end_time - start_time))
    return user, item



#attention
def attention_layer_dnn(K,input_vec):
    _initializer = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope("Attention_dnn",reuse=tf.AUTO_REUSE):
        att_w1 = tf.get_variable(name='att_w1', shape=[K,hidden_size], initializer=_initializer, trainable=True)
        att_b1 = tf.get_variable(name='att_b1', shape=[hidden_size],                initializer=_initializer, trainable=True)        
        att_w2 = tf.get_variable(name='att_w2', shape=[hidden_size,1],              initializer=_initializer, trainable=True)
        att_b2 = tf.get_variable(name='att_b2', shape=[1],                          initializer=_initializer, trainable=True)
        
        net_1 = tf.nn.sigmoid(tf.matmul(input_vec,att_w1) + att_b1)
        net_2 = tf.nn.sigmoid(tf.matmul(net_1,att_w2) + att_b2)
        return net_2

#用attention机制计算最终的user和item
def attention_dnn():
    #attention
    U_feature_input = tf.placeholder(tf.int64,[None,1])
    I_feature_input = tf.placeholder(tf.int64,[None,1])
    rating = tf.placeholder(tf.float32,[None,1])
    
    weight_user = []
    weight_item = []
    e={}
    for j in range(1,10):
        e['user_m%s'%j] = d['yelp_user_m%s'%j].astype(np.float32)
        e['user_m%s'%j] = tf.nn.embedding_lookup(e['user_m%s'%j], U_feature_input)
        e['user_m%s'%j] = tf.reshape(e['user_m%s'%j], [-1,K])
        weight_user.append(tf.exp(attention_layer_dnn(K=K,input_vec=e['user_m%s'%j])))


        e['item_m%s'%j] = (d['yelp_item_m%s'%j].T).astype(np.float32)
        e['item_m%s'%j] = tf.nn.embedding_lookup(e['item_m%s'%j], I_feature_input)
        e['item_m%s'%j] = tf.reshape(e['item_m%s'%j], [-1,K])
        weight_item.append(tf.exp(attention_layer_dnn(K=K,input_vec=e['item_m%s'%j])))

    if(merge == "attention"):
        user_K = 0
        for j in range(9):
            #这种语法居然真不报错
            user_K += weight_user[j]/np.sum(weight_user) * e['user_m%s'%(j+1)]
        item_K = 0
        for j in range(9):
            item_K += weight_item[j]/np.sum(weight_item) * e['item_m%s'%(j+1)]
    if(merge == "avg"):
        user_K = 0
        for j in range(9):
            #这种语法居然真不报错
            user_K += 1/9 * e['user_m%s'%(j+1)]
        item_K = 0
        for j in range(9):
            item_K += 1/9 * e['item_m%s'%(j+1)]
    if(merge == "concat"):
        user_K = e['user_m%s'%1]
        for j in range(2,10):
            user_K = tf.concat([user_K,e['user_m%s'%j]],1) 
        item_K = e['item_m%s'%1]
        for j in range(2,10):
            item_K = tf.concat([item_K,e['item_m%s'%j]],1) 

    return user_K, item_K, rating, U_feature_input, I_feature_input, weight_user, weight_item

#DNN
def DNN_layer(K,hidden_size,U,V):
    # user = user_feature(path_structure)
    # item = item_feature(path_structure)
    _initializer = tf.contrib.layers.xavier_initializer() 
    with tf.variable_scope("DNN",reuse=tf.AUTO_REUSE):
        input = tf.concat([U,V],1)
        if(merge == "concat"):
            w1 = tf.get_variable(name='w1', shape=[K*18,hidden_size], initializer=_initializer, trainable=True)
        else:
            w1 = tf.get_variable(name='w1', shape=[K*2,hidden_size], initializer=_initializer, trainable=True)
        b1 = tf.get_variable(name='b1', shape=[hidden_size],                  initializer=_initializer, trainable=True)
        w2 = tf.get_variable(name='w2', shape=[hidden_size,last_hidden_size], initializer=_initializer, trainable=True)
        b2 = tf.get_variable(name='b2', shape=[last_hidden_size],             initializer=_initializer, trainable=True)
        w3 = tf.get_variable(name='w3', shape=[last_hidden_size,1],           initializer=_initializer, trainable=True)
        b3 = tf.get_variable(name='b3', shape=[1],                            initializer=_initializer, trainable=True)
        
        net1 = tf.nn.relu(tf.matmul(input,w1) + b1)
        net2 = tf.nn.relu(tf.matmul(net1,w2) + b2)
        output = tf.nn.relu(tf.matmul(net2,w3) + b3)
        return output        

def RMSE_TF(rating,y_pred):
    rmse = np.sqrt(((rating-y_pred)**2).mean())
    return rmse

def MAE_TF(rating,y_pred):
    mae = np.mean(np.sqrt((rating-y_pred)**2))
    return mae

def train_dnn(epoches,batch_size,y_pred,loss,training_op,merged,writer,user_K, item_K, rating, U_feature_input, I_feature_input,weight_user,weight_item):  
    #tf.reset_default_graph()
    # training_op,loss,y_pred = train_loss(path_structure)
    #self.sess.run(tf.global_variables_initializer())
    print("Loading data...")
    start_time = time.time()
    #for path in path_structure:
    #UB = np.loadtxt('/home/zl/Desktop/Meta-graph实验/KSEM实验/amazon1/ub.txt')
    train_u,train_i,train_rating,test_u,test_i,test_rating =\
    yelp_train_user_id, yelp_train_item_id, yelp_train_star, yelp_test_user_id, yelp_test_item_id, yelp_test_star
    grosstime, grossrmse, grossmae = [], [], []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Start training...')
        for epoch in range(epoches):
            user_input,item_input,rating_ = shuffle(train_u,train_i,train_rating)
            for index in range(len(user_input)//batch_size + 1):
                batch_u = np.transpose([user_input[index*batch_size:(index+1)*batch_size]])
                batch_i = np.transpose([item_input[index*batch_size:(index+1)*batch_size]])
                batch_labels = np.transpose([rating_[index*batch_size:(index+1)*batch_size]])
                _,loss_train,rs = sess.run([training_op,loss,merged],
                        feed_dict = {U_feature_input:batch_u,I_feature_input:batch_i,rating:batch_labels})
                writer.add_summary(rs, epoch)
                format_str = '**%s:%d epoch,%d iteration, loss=%.4f'
                #print(format_str % (datetime.now(),epoch,index,loss_train))
            if(epoch%(epoches/100) == 0):
                print(epoch/(epoches/100),"%")    
        print("End training!")
         
        print('Start testing...')
        test_rmse = []
        test_mae = []
        for index in range(len(test_u)//batch_size + 1):
            batch_u = np.transpose([test_u[index*batch_size:(index+1)*batch_size]])
            batch_i = np.transpose([test_i[index*batch_size:(index+1)*batch_size]])
            batch_labels = np.transpose([test_rating[index*batch_size:(index+1)*batch_size]])
            y_pred_out = sess.run([y_pred],
                    feed_dict = {U_feature_input:batch_u,I_feature_input:batch_i,rating:batch_labels})
            rmse = RMSE_TF(y_pred_out, batch_labels)
            mae = MAE_TF(y_pred_out, batch_labels)
            #print(rmse)
            test_rmse.append(np.array(rmse).mean())
            test_mae.append(np.array(mae).mean())
        grossrmse.append(np.array(test_rmse).mean())
        grossmae.append(np.array(test_mae).mean())
        print('End testing!')
        end_time = time.time()
        #print('The gross time of %s is:'%rank,time.time()-t1)
        grosstime.append(end_time-start_time)
        
    print(grossrmse)
    print(grossmae)
    print(grosstime)

def DNN():
    #attention步，将M1-M7的每个user和item统一为一个user_K和item_K
    tf.reset_default_graph()
    user_K, item_K, rating, U_feature_input, I_feature_input, weight_user, weight_item = attention_dnn()

    print("Start DNN")
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config = config)
    #tf.global_variables_initializer().run()

    #####################################load_train_test_data#################################################

    #def train_loss(path_structure):
    with tf.name_scope('loss'):
        y_pred = DNN_layer(K,hidden_size,user_K,item_K)
        loss = tf.sqrt(tf.reduce_mean(tf.square(rating - y_pred)))
        training_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/att/", sess.graph) # tensorflow >=0.12
    sess.run(tf.global_variables_initializer())

    train_dnn(epoches,batch_size,y_pred,loss,training_op,merged,writer,user_K, item_K, rating, U_feature_input, I_feature_input,weight_user,weight_item)
    

def Bayesian_Knn():
    # start_time = time.time()
    # knn_bo_rmse = BayesianOptimization(
    #     Knn_Weight_RMSE,
    #     {
    #     'w1':(0,1),
    #     'w2':(0,1),
    #     'w3':(0,1),
    #     'w4':(0,1),
    #     'w5':(0,1),
    #     'w6':(0,1),
    #     'w7':(0,1),
    #     'w8':(0,1),
    #     'w9':(0,1),
    #     }
    #     )
    # knn_bo_rmse.maximize()

    # index_rmse = []
    # for i in knn_bo_rmse.res:
    #     index_rmse.append(i['target'])
    # max_index_rmse = index_rmse.index(max(index_rmse))
    # #print(index_rmse)
    # print("RMSE:",-index_rmse[max_index_rmse])
    # end_time = time.time()
    # print("Time: %f s"%(end_time - start_time))

    start_time = time.time()
    knn_bo_mae = BayesianOptimization(
        Knn_Weight_MAE,
        {
        'w1':(0,1),
        'w2':(0,1),
        'w3':(0,1),
        'w4':(0,1),
        'w5':(0,1),
        'w6':(0,1),
        'w7':(0,1),
        'w8':(0,1),
        'w9':(0,1),
        }
        )
    knn_bo_mae.maximize()

    index_mae = []
    for i in knn_bo_mae.res:
        index_mae.append(i['target'])
    max_index_mae = index_mae.index(max(index_mae))
    #print(index_mae)
    print("MAE:",-index_mae[max_index_mae])
    end_time = time.time()
    print("Time: %f s"%(end_time - start_time))

def Knn_Weight_RMSE(w1,w2,w3,w4,w5,w6,w7,w8,w9):
    user = 0
    item = 0
    w = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
    for j in range(9):
        user += w[j]*d['yelp_user_m%s'%(j+1)]
        item += w[j]*d['yelp_item_m%s'%(j+1)]
    item = item.T
    user_train = user[yelp_train_user_id]
    user_test  = user[yelp_test_user_id]
    item_train = item[yelp_train_item_id]
    item_test  = item[yelp_test_item_id]
    X_train = np.concatenate((user_train,item_train),axis=1)
    X_test = np.concatenate((user_test,item_test),axis=1)
    y_train = yelp_train_star
    y_test  = yelp_test_star
    y_predict = []
    nearest = 10
    knn = KNeighborsClassifier(n_neighbors=nearest,algorithm='kd_tree',leaf_size=30) 
    knn.fit(X_train,y_train)
    for i in range(np.shape(X_test)[0]):   
            distance, number=knn.kneighbors(X_test[i].reshape(1, -1),nearest,True)
            distance = distance.flatten()
            number = number.flatten()
            y = 0
            for j in range(nearest):
                y += y_train[number[j]] * 1.0 / nearest
            y_predict.append(y)
    rmse = RMSE(y_predict,y_test)
    return -rmse

def Knn_Weight_MAE(w1,w2,w3,w4,w5,w6,w7,w8,w9):
    user = 0
    item = 0
    w = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
    for j in range(9):
        user += w[j]*d['yelp_user_m%s'%(j+1)]
        item += w[j]*d['yelp_item_m%s'%(j+1)]
    item = item.T
    user_train = user[yelp_train_user_id]
    user_test  = user[yelp_test_user_id]
    item_train = item[yelp_train_item_id]
    item_test  = item[yelp_test_item_id]
    X_train = np.concatenate((user_train,item_train),axis=1)
    X_test = np.concatenate((user_test,item_test),axis=1)
    y_train = yelp_train_star
    y_test  = yelp_test_star
    y_predict = []
    nearest = 10
    knn = KNeighborsClassifier(n_neighbors=nearest,algorithm='kd_tree',leaf_size=30) 
    knn.fit(X_train,y_train)
    for i in range(np.shape(X_test)[0]):   
            distance, number=knn.kneighbors(X_test[i].reshape(1, -1),nearest,True)
            distance = distance.flatten()
            number = number.flatten()
            y = 0
            for j in range(nearest):
                y += y_train[number[j]] * 1.0 / nearest
            y_predict.append(y)
    mae = MAE(y_predict,y_test)
    return -mae

#先写个PMF压压惊
def PMF():
    start_time = time.time()
    y_predict = []
    yelp_user_m1 = np.loadtxt(b+'/yelp_%s_%s/yelp_user_m%s.txt'%(n,K,1))
    yelp_item_m1 = np.loadtxt(b+'/yelp_%s_%s/yelp_item_m%s.txt'%(n,K,1))
    for j in range(len(yelp_test_number)):
        user = yelp_user_m1[yelp_test_user_id[j],0:10]
        item = yelp_item_m1[0:10,yelp_test_item_id[j]]
        y = np.dot(user,item)
        #y = int(y)
        y_predict.append(y)
    y_predict = np.array(y_predict)
    y_test  = yelp_test_star
    RMSE_PMF = RMSE(y_test=y_test, y_predict=y_predict)
    MAE_PMF = MAE(y_test=y_test, y_predict=y_predict)
    end_time = time.time()
    print("RMSE_PMF:", RMSE_PMF)
    print("MAE_PMF:", MAE_PMF)
    print("Time: %f s"%(end_time - start_time))

def RMSE(y_test, y_predict):
    if(np.shape(y_test) != np.shape(y_predict)):
        print("RMSE is going wrong!")
        return -1;
    else:
        y = y_test - y_predict
        result = np.linalg.norm(y,ord=2) / math.sqrt(len(y_test))
        return result

def MAE(y_test, y_predict):
    if(np.shape(y_test) != np.shape(y_predict)):
        print("MAE is going wrong!")
        return -1;
    else:
        y = y_test - y_predict
        result = np.linalg.norm(y,ord=1) / len(y_test)
        return result

if __name__=="__main__":
    
    #设置参数
    n, m, K, learning_rate, batch_size, epoches,\
    hidden_size, last_hidden_size, train_ratio, merge = settings()

    b = os.getcwd()
    if(not os.path.exists(b+'/yelp_%s_%s/integrality'%(n,K))):
        if(not os.path.exists(b+'/yelp_%s_%s'%(n,K))):
            os.mkdir(b+'/yelp_%s_%s'%(n,K))

        #数据集加载
        data_yelp_business = read_yelp('business')
        data_yelp_review = read_yelp('review')
        data_yelp_user = read_yelp('user')
        #data_yelp_tip = read_yelp('tip')
        #data_yelp_checkin = read_yelp('checkin')
        
        #数据编号
        yelp_user_id = collect_yelp('user')
        yelp_item_id = collect_yelp('business')
        
        #补齐长度
        yelp_user_id, yelp_item_id = samelength(A=yelp_user_id, B=yelp_item_id)
        
        #划分训练集和测试集 #这个也需要编号!
        yelp_train_number, yelp_test_number = train_test_split(np.arange(n), test_size=0.3, random_state=int("0801"))
        
        data_yelp_review_train, data_yelp_review_test =\
        np.array(data_yelp_review)[yelp_train_number], np.array(data_yelp_review)[yelp_test_number] 
        
        #把训练集和测试集编号化
        yelp_train_user_id, yelp_train_item_id, yelp_test_user_id, yelp_test_item_id,\
        yelp_train_star, yelp_test_star = numeralization_yelp()
        
        #元图 Meta-graphs d为装载所有中间变量的字典
        d = {}
        d['yelp_M%s'%1], d['yelp_M%s_norating'%1] = locals()['assemble_yelp_M%s'%1]()
        for j in range(1,10): 
            if((not os.path.exists(b+'/yelp_%s_%s/yelp_user_m%s.txt'%(n,K,j))) or\
            (not os.path.exists(b+'/yelp_%s_%s/yelp_item_m%s.txt'%(n,K,j)))):
                d['yelp_M%s'%j], d['yelp_M%s_norating'%j] = locals()['assemble_yelp_M%s'%j]()
                d['yelp_user_m%s'%j], d['yelp_item_m%s'%j] = lfm_sparse(d['yelp_M%s'%j])
                np.savetxt(b+'/yelp_%s_%s/yelp_user_m%s.txt'%(n,K,j),d['yelp_user_m%s'%j],fmt='%.8f') 
                np.savetxt(b+'/yelp_%s_%s/yelp_item_m%s.txt'%(n,K,j),d['yelp_item_m%s'%j],fmt='%.8f') 
            else:
                d['yelp_user_m%s'%j] = np.loadtxt(b+'/yelp_%s_%s/yelp_user_m%s.txt'%(n,K,j))
                d['yelp_item_m%s'%j] = np.loadtxt(b+'/yelp_%s_%s/yelp_item_m%s.txt'%(n,K,j))

        #存储数据
        np.savetxt(b+'/yelp_%s_%s/yelp_train_number.txt'%(n,K),yelp_train_number,fmt='%d')
        np.savetxt(b+'/yelp_%s_%s/yelp_test_number.txt'%(n,K),yelp_test_number,fmt='%d')
        np.savetxt(b+'/yelp_%s_%s/yelp_train_user_id.txt'%(n,K),yelp_train_user_id,fmt='%d')
        np.savetxt(b+'/yelp_%s_%s/yelp_train_item_id.txt'%(n,K),yelp_train_item_id,fmt='%d')
        np.savetxt(b+'/yelp_%s_%s/yelp_test_user_id.txt'%(n,K),yelp_test_user_id,fmt='%d')
        np.savetxt(b+'/yelp_%s_%s/yelp_test_item_id.txt'%(n,K),yelp_test_item_id,fmt='%d')
        np.savetxt(b+'/yelp_%s_%s/yelp_train_star.txt'%(n,K),yelp_train_star,fmt='%.1f')
        np.savetxt(b+'/yelp_%s_%s/yelp_test_star.txt'%(n,K),yelp_test_star,fmt='%.1f')

        os.mkdir(b+'/yelp_%s_%s/integrality'%(n,K))
   
    #数据加载
    yelp_train_number  = np.loadtxt(b+'/yelp_%s_%s/yelp_train_number.txt'%(n,K),dtype=int)
    yelp_test_number   = np.loadtxt(b+'/yelp_%s_%s/yelp_test_number.txt'%(n,K),dtype=int)
    yelp_train_user_id = np.loadtxt(b+'/yelp_%s_%s/yelp_train_user_id.txt'%(n,K),dtype=int)
    yelp_train_item_id = np.loadtxt(b+'/yelp_%s_%s/yelp_train_item_id.txt'%(n,K),dtype=int)
    yelp_test_user_id  = np.loadtxt(b+'/yelp_%s_%s/yelp_test_user_id.txt'%(n,K),dtype=int)
    yelp_test_item_id  = np.loadtxt(b+'/yelp_%s_%s/yelp_test_item_id.txt'%(n,K),dtype=int)
    yelp_train_star    = np.loadtxt(b+'/yelp_%s_%s/yelp_train_star.txt'%(n,K),dtype=int)
    yelp_test_star     = np.loadtxt(b+'/yelp_%s_%s/yelp_test_star.txt'%(n,K),dtype=int)
    d = {}
    for j in range(1,10):
        d['yelp_user_m%s'%j] = np.loadtxt(b+'/yelp_%s_%s/yelp_user_m%s.txt'%(n,K,j))
        d['yelp_item_m%s'%j] = np.loadtxt(b+'/yelp_%s_%s/yelp_item_m%s.txt'%(n,K,j))      
    
    #PMF()
    #DNN()
    Bayesian_Knn()