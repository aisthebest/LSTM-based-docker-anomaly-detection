#coding=utf-8
##建模和测试主体程序
import sys
import getopt
import signal
import time
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)##array全部展示，无省略号
from functools import reduce
import copy
import threading
import multiprocessing

# 定义TensorFlow配置
config = tf.ConfigProto()
# 配置GPU内存分配方式，按需增长，很关键
config.gpu_options.allow_growth = True
# 配置可使用的显存比例
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# 在创建session的时候把config作为参数传进去
sess = tf.Session(config = config)

class lstmMain:
    def __init__(self,input_size=1,output_size=1,time_step=15,rnn_unit=100,lstm_layers=2,batch_size=100,lr=0.005,epoch_num=100,model_name='lstm_model',state_window=5,state_threshold=1,frame_window=20,abs_threshold=2,ab0_threshold=5,abnormal_threshold=3):
        self.input_size=input_size
        self.output_size=output_size
        self.time_step=time_step
        self.rnn_unit=rnn_unit
        self.lstm_layers=lstm_layers
        self.batch_size=batch_size
        self.lr=lr
        self.epoch_num=epoch_num
        self.model_name=model_name
        self.state_window=state_window
        self.state_threshold=state_threshold
        self.frame_window=frame_window
        self.abs_threshold=abs_threshold
        self.ab0_threshold=ab0_threshold
        self.abnormal_threshold=abnormal_threshold

    def gcd_main(self,a,b):
        if b==0:
            return a
        return self.gcd_main(b,a%b)

    def gcd(self,frequency):
        gcd_v=1
        for n,f in enumerate(frequency):
            if n==0:
                gcd_v=f
            else:
                gcd_v=self.gcd_main(gcd_v,f)
        return gcd_v

    ##格式建模数据，同比缩减训练样本
    def get_train_data(self,data_set_list):
        batch_index=[]##一个训练文件的batch索引
        train_x=[]
        train_y=[]
        diff_tmp=[]##每一个文件夹下多个文件，多个文件的所有不同样本
        map_ratio={}##保存{time_step:{next value:frequency,next value:frequency,...},time_step:{next value:frequency,next value:frequency,...}}最后按照比例在diff_tmp中添加相应样本数量
        ##首先获取所有建模数据的不相同短序列，和所有映射不同值的次数
        for data_index, data_set_name in enumerate(data_set_list):
            fd=open(data_set_name)
            content=fd.read().split('\n')
            fd.close()
            if(len(content) < self.time_step+1):##若文件内不够一个序列，则跳过该文件
                continue
            sample=[]
            diff=[]
            for n,i in enumerate(content):
                if i!='':
                    sample.append(int(i))
                    if len(sample)==self.time_step+1:
                        if str(sample[0:-1]) not in map_ratio:
                            map_ratio[str(sample[0:-1])]={}
                            map_ratio[str(sample[0:-1])][str(sample[-1])]=1
                        else:
                            if str(sample[-1]) not in map_ratio[str(sample[0:-1])]:
                                map_ratio[str(sample[0:-1])][str(sample[-1])]=1
                            else:
                                map_ratio[str(sample[0:-1])][str(sample[-1])]+=1

                        if sample not in diff_tmp:
                            diff_tmp.append(copy.deepcopy(sample))##保存训练该模型的所有不同样本
                        del sample[0]
        # print('diff_tmp',diff_tmp,'map_ratio',map_ratio)##包含了所有不相同短序列，所有映射不同值的次数
        ##根据映射值比例，同比增加样本数
        for time_step_value in map_ratio:
            if len(map_ratio[time_step_value])!=1:##一个time_step不止映射一个值，即存在12-3和12-4，需要按照比例在diff_tmp中增加样本。如果只映射一个值，那diff_tmpo样本中只有一个，大幅减少样本数量
                frequency=[]
                for next_value in map_ratio[time_step_value]:
                    next_value_frequency=map_ratio[time_step_value][next_value]
                    frequency.append(int(next_value_frequency))
                gcd_v=self.gcd(frequency)##至少两个值，需要计算最大公约数，然后求得每个样本应该增加多少个
                for next_value in map_ratio[time_step_value]:
                    next_value_frequency=map_ratio[time_step_value][next_value]
                    count=next_value_frequency/gcd_v##同比减少,在最后使用的样本中，需要有count个
                    add_sample=[]
                    time_step_int=[]
                    for i in time_step_value[1:-1].split(' '):##存入map_ratio时是str，所以需要由'[12,2,3]'转为[12,2,3]
                        time_step_int.append(int(i.strip(',')))
                    add_sample.extend(time_step_int)
                    add_sample.append(int(next_value))
                    for i in range(int(count)):
                        if i!=count-1:
                            diff_tmp.append(add_sample)
        # print('diff_tmp',diff_tmp，'len(diff_tmp)',len(diff_tmp))##包含了按照比例处理数据后的所有短序列
        ##生成batch，处理为one-hot用于网络输入
        for seq in diff_tmp:
            data=np.array(seq).reshape(-1,1)
            x=data[0:-1]
            train_x.append(x.tolist())
            y=data[-1]
            train_y.append(y.tolist())
        if len(train_x)==0:
            return batch_index,train_x,train_y##不够建模       
        for i in range(len(train_x)+1):
            if i % self.batch_size==0:
                batch_index.append(i)
        if batch_index[-1]!=len(train_x):##把最后不够batch_size的数据加上，不漏掉训练数据
            batch_index.append(len(train_x))
        train_x=np.asarray(tf.keras.utils.to_categorical(train_x,self.input_size)).reshape(-1,self.time_step,self.input_size)##tf1.4上要加这个才能每time_step个为一个输入
        return batch_index,train_x,np.asarray(tf.keras.utils.to_categorical(train_y,self.output_size))
        # return batch_index,np.asarray(tf.keras.utils.to_categorical(train_x,input_size)), np.asarray(tf.keras.utils.to_categorical(train_y,output_size))##tf1.11

    ##未同比去重的建模数据，用于确定状态异常阈值和校正库
    def get_train_test_data(self,data_set_list):
        batch_index=[]##一个训练文件的batch索引
        train_x=[]
        train_y=[]
        for data_index, data_set_name in enumerate(data_set_list):
            command="du -h %s | awk '{print $1}'" % (data_set_name)
            fp=os.popen(command,"r")
            ret=fp.read().strip('\n')
            fp.close()
            if ret=='0':##该数据文件没内容，大小为0k，如果不处理就read_csv，会报错pd
                continue
            ##原不对重复样本处理，直接读取文件内容
            df=pd.read_csv(data_set_name,header=None)
            data=np.array(df)
            if(len(data) < self.time_step+1):##若文件内不够一个序列，则跳过该文件
                continue
            data_train=data[:]
            for i in range(len(data_train)-self.time_step):##不+1，最后一个time_step没有预测值，舍弃
                x=data_train[i:i+self.time_step]
                train_x.append(x.tolist())
                y=data_train[i+self.time_step]
                train_y.append(y.tolist())       
        for i in range(len(train_x)+1):##如果一次训练数据过多，在确定阈值时全部加载进内存会报错无法分配那么大内存，所以还是要分batch加入
            if i % self.batch_size==0:
                batch_index.append(i)
        if batch_index[-1]!=len(train_x):##把最后不够batch_size的数据加上，不漏掉数据
            batch_index.append(len(train_x))        
        train_x=np.asarray(tf.keras.utils.to_categorical(train_x,self.input_size)).reshape(-1,self.time_step,self.input_size)
        return batch_index,train_x,np.asarray(tf.keras.utils.to_categorical(train_y,self.output_size))
        # return batch_index,np.asarray(tf.keras.utils.to_categorical(train_x,input_size)), np.asarray(tf.keras.utils.to_categorical(train_y,output_size))

    ##定义神经网络变量：输入层、输出层权重、偏置、dropout参数
    weights={}
    biases={}
    keep_prob=1
    ##由于output_size变化，且该参数需要保存到模型，所以这里重新设置值
    def set_para(self):
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.weights={
             'in':tf.Variable(tf.random_normal([self.input_size,self.rnn_unit]),name="w_in"),
             'out':tf.Variable(tf.random_normal([self.rnn_unit,self.output_size]),name="w_out")
            }
        self.biases={
                'in':tf.Variable(tf.constant(0.1,shape=[self.rnn_unit,]),name="b_in"),
                'out':tf.Variable(tf.constant(0.1,shape=[self.output_size,]),name="b_out")
            }  

    #定义lstm单元
    def lstmCell(self):
        basicLstm = tf.nn.rnn_cell.LSTMCell(self.rnn_unit)
        drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=self.keep_prob)
        return basicLstm

    ##定义网络结构
    def lstm(self,X):
        batch_size=tf.shape(X)[0]
        time_step=tf.shape(X)[1]
        w_in=self.weights['in']
        b_in=self.biases['in']
        input=tf.reshape(X,[-1,self.input_size])
        input_rnn=tf.matmul(input,w_in)+b_in
        input_rnn=tf.reshape(input_rnn,[-1,time_step,self.rnn_unit])
        cell = tf.nn.rnn_cell.MultiRNNCell([self.lstmCell() for i in range(self.lstm_layers)])
        init_state=cell.zero_state(batch_size,dtype=tf.float32)
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)
        w_out=self.weights['out']
        b_out=self.biases['out']
        output=tf.reshape(output_rnn[:,-1],[-1,self.rnn_unit])##取每个time_step的最后一个时刻输出
        pred=tf.matmul(output,w_out)+b_out##网络输出，为每个系统调用的概率
        return final_states,pred

    ##训练模型
    vb=""
    def train_lstm(self,data_set_list,p_modelname,batch_index,train_x,train_y):
        # print(("===train model parameters:output_size=%d,time_step=%d,rnn_unit=%d,lstm_layers=%d,batch_size=%d,lr=%f,epoch_num=%d,model_name=%s,state_window=%d")\
        #         % (self.output_size,self.time_step,self.rnn_unit,self.lstm_layers,self.batch_size,self.lr,self.epoch_num,self.model_name,self.state_window))
        global config
        vb=p_modelname
        self.vb=vb##确定状态阈值时需要使用
        X=tf.placeholder(tf.float32, shape=[None,self.time_step,self.input_size])
        Y=tf.placeholder(tf.float32, shape=[None,self.output_size])
        print('train sample len',len(train_x))##同比去重后用于模型训练的样本个数
        with tf.variable_scope(vb):
            _,pred=self.lstm(X)
        # cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=Y)
        cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=Y)
        loss=tf.reduce_mean(cross_entropy)
        train_op=tf.train.AdamOptimizer(self.lr).minimize(loss)##一个batch的loss
        saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
        lossp=[]
        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.epoch_num):
                index=len(train_x)##shuffle
                index=np.random.permutation(index)
                train_x_=train_x[index]##将index条time_step打乱顺序(time_step内不变，对应标签不变)
                train_y_=train_y[index]

                for step in range(len(batch_index)-1):
                    _,loss_=sess.run([train_op,loss],feed_dict={X:train_x_[batch_index[step]:batch_index[step+1]],Y:train_y_[batch_index[step]:batch_index[step+1]],self.keep_prob:0.5})
                    lossp.append(loss_)##每个epoch的每个batch都画入图中
                print("Number of epoch:%d, loss:%f" % (i,loss_))##每个epoch输出

                # # if i%10==0 or i == self.epoch_num-1:##虽然可通过该方式看到准确率，但此处会重复添加节点使得模型大于2G无法保存，所以只能可视确定epoch，若爆掉保存不了模型，则需注释
                # if i==self.epoch_num-1:
                #     batch_index,train_x,train_y=self.get_train_test_data(data_set_list)
                #     # print('original sample len',p_modelname,len(train_x))
                #     cp_value=[]
                #     for step in range(len(batch_index)-1):##不能一下全部加载，当训练样本多时会超出内存
                #         prob=sess.run(pred,feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],self.keep_prob:1})
                #         y_predict=tf.nn.softmax(prob)
                #         correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(train_y[batch_index[step]:batch_index[step+1]],1))
                #         cp_value.extend(tf.cast(correct_prediction, "float").eval())
                #     accuracy = tf.reduce_mean(cp_value)##正确/总数
                #     test_acc=accuracy.eval()
                #     print ('iteration %d: loss %.6f accuracy %.6f' % (i,loss_,test_acc))

            ##将loss输出成图
            plt.figure()
            plt.plot(list(range(len(lossp))), lossp, color='b')
            if os.path.exists(self.model_name+'_train_loss')!=1:##不存在则新建,否则直接覆盖
                os.mkdir(self.model_name+'_train_loss')
            plt.savefig("./"+self.model_name+"_train_loss/"+p_modelname+"_train_loss.png")##包含所有epoch的每个batch的loss
            print("The data_set used for training:%s\nTotal num:%d" % (data_set_list,len(train_y)))##输出用于建立该模型的所有数据，以及训练样本数
            
            ##保存模型
            if os.path.exists(self.model_name)!=1:##不存在则新建,否则直接覆盖
                os.mkdir(self.model_name)
            if os.path.exists(self.model_name+'/'+p_modelname)!=1:
                os.mkdir(self.model_name+'/'+p_modelname)
            print("Model_save:%s" % (saver.save(sess,self.model_name+'/'+p_modelname+'/model.ckpt')))

    ##模型已建立好，针对建模数据确定状态异常阈值，用于检测阶段
    def set_state_threshold(self,data_set_list,p_modelname):
        X=tf.placeholder(tf.float32, shape=[None,self.time_step,self.input_size])
        Y=tf.placeholder(tf.float32, shape=[None,self.output_size])
        batch_index,train_x,train_y=self.get_train_test_data(data_set_list)
        print("original sample len:",len(train_x))##未去重处理的原始训练样本数量
        with tf.variable_scope(self.vb,reuse=tf.AUTO_REUSE):
            _,pred=self.lstm(X)
        saver=tf.train.Saver(tf.global_variables())
        revise={}
        with tf.Session() as sess:
            module_file = tf.train.latest_checkpoint(self.model_name+'/'+p_modelname)
            saver.restore(sess, module_file)
            for step in range(len(batch_index)-1):##不能一下全部加载，当训练样本多时会超出内存
                prob=sess.run(pred,feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],self.keep_prob:1})
                y_predict=tf.nn.softmax(prob)
                correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(train_y[batch_index[step]:batch_index[step+1]],1))##预测下一个和真实下一个相同则1，不同则0

                ##提取每个预测值的概率，将状态窗口state_window内概率连乘，滑窗1，取最小值，以此作为该进程的检测阈值
                predict_prob_list=[]
                predict=y_predict.eval()
                for i,one_y in enumerate(train_y[batch_index[step]:batch_index[step+1]]):
                    true_next=np.argmax(one_y)
                    predict_prob=predict[i][true_next]##预测为此真实值的概率
                    predict_prob_list.append(predict_prob)
                    if len(predict_prob_list)==self.state_window:##滑窗1，计算state_window内此状态出现概率，确定阈值
                        _t=reduce(lambda x,y:x*y,predict_prob_list)
                        if self.state_threshold-_t>=0.0000001:
                            self.state_threshold=_t##取最小概率做该进程的阈值
                        del predict_prob_list[0]
                # print('-----state_threshold:',state_threshold)

                ##模型校正，判错的真实值：前序time_step。将accuracy更准确
                correct=correct_prediction.eval().astype(int)
                index_0=[i for i,x in enumerate(correct) if x == 0]
                for i,index in enumerate(index_0):
                    key=str(np.argmax(train_y[batch_index[step]+index]))
                    tmp=[]
                    for i in train_x[batch_index[step]+index]:
                        tmp.append(np.argmax(i))
                    value=str(tmp)
                    if key not in revise:
                        revise[key]=[]
                    if value not in revise[key]:
                        revise[key].append(value)

                batch_index[step+1]=batch_index[step+1]-self.state_window-1 ##不漏掉batch之间的状态
            
            if len(revise)!=0:
                if os.path.exists(self.model_name+'/'+p_modelname+'/revise'):##校正库加入模型
                    os.remove(self.model_name+'/'+p_modelname+'/revise')
                fp = open(self.model_name+'/'+p_modelname+'/revise','w')
                fp.write(str(revise))
                fp.close()
            
            print('-----state_threshold:',self.state_threshold)

    #格式测试数据
    def get_test_data(self,data):
        data_test=data[:]
        test_x=[]
        test_y=[]
        for i in range(len(data_test)-self.time_step):##不加1
            x=data_test[i:i+self.time_step]
            test_x.append(x.tolist())
            test_y.extend(data_test[i+self.time_step])
        test_x=np.asarray(tf.keras.utils.to_categorical(test_x,self.input_size)).reshape(-1,self.time_step,self.input_size)
        return test_x,np.asarray(tf.keras.utils.to_categorical(test_y,self.output_size))
        # return np.asarray(tf.keras.utils.to_categorical(test_x,input_size)),np.asarray(tf.keras.utils.to_categorical(test_y,output_size))

    ##测试模型
    def test(self,data,data_set_name,vb):
        normal=0##最后返回该值，如果是1表明该测试数据在使用该模型时是正常的
        abnormal=0
        X=tf.placeholder(tf.float32, shape=[None,self.time_step,self.input_size])
        test_x,test_y=self.get_test_data(data)
        with tf.variable_scope(vb,reuse=tf.AUTO_REUSE):
            _,pred=self.lstm(X)
        saver=tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            #参数恢复
            module_file = tf.train.latest_checkpoint(self.model_name)
            saver.restore(sess, module_file)        
            prob=sess.run(pred,feed_dict={X:test_x[:],self.keep_prob:1})
            y_predict=tf.nn.softmax(prob)
            
            ##判定方法1：状态概率判定。提取每个预测值的概率，滑窗1，将一个局部帧窗口内的状态窗口state_window内概率连乘，局部帧窗口内有abs_threshold个小于检测阈值的，则此局部帧内异常输出
            frame_index=0        
            abnormal_num=0
            predict_prob_list=[]
            true_num_list=[]
            abnormal_tmp={}
            abs_all={}
            
            predict=y_predict.eval()
            for i,one_y in enumerate(test_y):
                true_next=np.argmax(one_y)
                true_num_list.append(true_next)
                predict_prob=predict[i][true_next]##预测下一个值的概率
                predict_prob_list.append(predict_prob)
                if len(predict_prob_list) == self.state_window:##滑窗1
                    frame_index += 1
                    if frame_index == self.frame_window+1:
                        frame_index=1
                        abnormal_tmp.clear()
                        abnormal_num=0
                    _t=reduce(lambda x,y:x*y,predict_prob_list)
                    if self.state_threshold-_t>=0.0000001:##此状态窗口小于检测阈值
                        if str(true_num_list) not in abnormal_tmp:##状态窗口异常，看后续局部帧内是否有超过检测阈值数量个异常，若有则输出帧内异常状态
                            abnormal_tmp[str(true_num_list)]=1
                        else:
                            abnormal_tmp[str(true_num_list)] += 1
                        abnormal_num += 1
                        if abnormal_num == self.abs_threshold:##此局部帧内已经有足够的异常数量，则警报一次，将该帧内异常的那几个子序列加入总异常序列中,再从其后开始算新帧，避免重复输出
                            _all=abs_all.copy()
                            _all.update(abnormal_tmp)##添加了新值，且相同键的值覆盖了
                            for _a in abnormal_tmp:
                                if _a in abs_all:
                                    _all[_a] += abs_all[_a]
                            abs_all.update(_all)
                            abnormal_tmp.clear()
                            abnormal_num=0
                            frame_index=0
                    del predict_prob_list[0]
                    del true_num_list[0]
            if len(abs_all)!=0:
                abs_all_nums=reduce(lambda x,y:x+y,abs_all.values())
                if abs_all_nums/self.abs_threshold >= self.abnormal_threshold:
                    abnormal=1
                    # print('---FIRST:Abnormal process:%s' % (data_set_name))
                    # print('abs_all:%s' % (abs_all))
                    # print('---FIRST:Abnormal process:%s\nTotal nums:%d Total ab types:%d Total ab state nums:%d' % (data_set_name,len(test_y),len(abs_all),abs_all_nums/self.abs_threshold))
            
            ##判定方法2：若概率方式判定正常，则进行异常0判定。
            # abnormal=0
            if abnormal==0:
                ##以模型预测原建模数据的错误数据库来校正测试时同样的错误0
                correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(test_y,1))##预测下一个和真实下一个相同则1，不同则0
                correct=correct_prediction.eval().astype(int) 
                # print('no revise\n%s' % (correct))
                if os.path.exists(self.model_name+'/revise'):
                    revise={}
                    size = os.stat(self.model_name+'/revise').st_size
                    fp = open(self.model_name+'/revise', 'rb')
                    fp.seek(0, 0)
                    revise = eval(fp.read(size))
                    fp.close()
                    index_0=[i for i,x in enumerate(correct) if x == 0]
                    for i,index in enumerate(index_0):
                        true_key=str(np.argmax(test_y[index]))
                        tmp=[]
                        for i in test_x[index]:
                            tmp.append(np.argmax(i))
                        true_value=str(tmp)
                        if true_key in revise and true_value in revise[true_key]:
                            correct[index]=1
                    # print('after revise\n%s' % (correct))

                ##校正后，如果异常0在局部帧内出现了ab0_threshold个，则从帧内第一个0到帧内阈值个0处的真实值输出作为异常短序列
                index_0=[i for i,x in enumerate(correct) if x == 0]
                start=0
                index=1
                count=1
                ab0_seq=[]
                ab0_all={}
                while start < len(index_0)-self.ab0_threshold+1:##就算最后不足一个state窗口，只要最后ab0_threshold个异常0可能在一个state窗口内，都需要记录警报
                    if index_0[start+index]<index_0[start]+self.frame_window:
                        count+=1
                        if count == self.ab0_threshold:
                            for _i in range(index_0[start],index_0[start+index]+1):
                                ab0_seq.append(np.argmax(test_y[_i]))##此异常短序列不定长，只要局部帧窗口内有阈值个0，就从第一个0到阈值个处都输出，再从阈值后那个0开始计算
                            if str(ab0_seq) not in ab0_all:##存入总的异常子序列中
                                ab0_all[str(ab0_seq)]=1
                            else:
                                ab0_all[str(ab0_seq)] += 1
                            start+=index+1##从阈值个数后开始重新局部帧计数
                            index=1
                            count=1
                            ab0_seq[:]=[]
                        else:
                            index+=1
                    else:
                        start+=index
                        index=1
                        count=1
                if len(ab0_all)!=0:
                    ab0_all_nums=reduce(lambda x,y:x+y,ab0_all.values())
                    if ab0_all_nums>=self.abnormal_threshold:
                        abnormal=1
                        # print('---SECOND:Abnormal process:%s' % (data_set_name))
                        # print('ab0_all:%s' % (ab0_all))
                        # print('---SECOND:Abnormal process:%s\nTotal nums:%d Total ab types:%d Total ab 0 nums:%d' % (data_set_name,len(test_y),len(ab0_all),ab0_all_nums))
                    else:
                        normal=1
                        # print('Normal process:%s---Relative to model:%s' % (data_set_name,self.model_name))
                else:
                    normal=1
                    # print('Normal process:%s---Relative to model:%s' % (data_set_name,self.model_name))
        return normal

def train_data_main(input_size,output_size,time_step,rnn_unit,lstm_layers,batch_size,lr,epoch_num,model_name,state_window,data_set_list):
    if len(data_set_list)==1:
        p_modelname=data_set_list[0].split('/')[-1].split('.')[0]##模型参数域名。小模型name，也就是建模数据name，之后可以契合采集时按照进程name命名数据文件。小模型全部存到大模型文件夹内
    else:
        p_modelname=data_set_list[0].split('/')[-2]
    p=lstmMain(input_size,output_size,time_step,rnn_unit,lstm_layers,batch_size,lr,epoch_num,model_name,state_window)
    
    start_time=time.time()
    batch_index,train_x,train_y=p.get_train_data(data_set_list)
    if len(train_x)==0:##输入文件的数据量不够建模，则跳过该数据，建立下一个模型
        print('The data is not enough for train a model',data_set_list)
        return
    end_time=time.time()
    print("Load data costs:%ss" % (end_time-start_time))
    
    start_time=time.time()
    tf.reset_default_graph()
    p.set_para()##设置权重等参数    
    p.train_lstm(data_set_list,p_modelname,batch_index,train_x,train_y)
    end_time=time.time()
    print("Train model costs:%ss" % (end_time-start_time))
    
    ##用训练好的模型测试建模数据，得到状态异常阈值state_threshold写入模型设置
    start_time=time.time()
    p.set_state_threshold(data_set_list,p_modelname)
    end_time=time.time()
    print("Set state threshold costs:%ss" % (end_time-start_time))
    
    ##将模型设置写入，待测试时使用
    model_setting={'output_size':p.output_size,'time_step':p.time_step,'rnn_unit':p.rnn_unit,'lstm_layers':p.lstm_layers,'state_window':p.state_window,'state_threshold':p.state_threshold,'model_vb':p_modelname}
    if os.path.exists(p.model_name+'/'+p_modelname+'/model_setting'):
        os.remove(p.model_name+'/'+p_modelname+'/model_setting')
    fp = open(p.model_name+'/'+p_modelname+'/model_setting','w')
    fp.write(str(model_setting))
    fp.close()

def test_data_main(model_list,data_set_name,abnormal_threshold,frame_window,abs_threshold,ab0_threshold,cover_st):
    df=pd.read_csv(data_set_name,header=None)
    data=np.array(df)
    result=0
    for i in model_list:
        p=lstmMain(model_name=i,abnormal_threshold=abnormal_threshold,frame_window=frame_window,abs_threshold=abs_threshold,ab0_threshold=ab0_threshold)
        ##从所使用的模型model_name设置中读取该模型参数。其余参数frame_window,abs_threshold,ab0_threshold由输入确定，或采用默认
        size=os.stat(p.model_name+'/model_setting').st_size
        fp=open(p.model_name+'/model_setting', 'rb')
        fp.seek(0, 0)
        model_setting = eval(fp.read(size))
        fp.close()
        p.output_size=model_setting['output_size']
        p.time_step=model_setting['time_step']
        p.rnn_unit=model_setting['rnn_unit']
        p.lstm_layers=model_setting['lstm_layers']
        p.state_window=model_setting['state_window']
        p.state_threshold=model_setting['state_threshold']##从模型设置中读入状态阈值，用于测试
        if cover_st!=0:##如果命令行指定了状态阈值，则不用模型中保存的值（因该值可能会因为建模数据只一次运行而太拟合，导致阈值很低，误报大，或者训练数据太乱，阈值太低，检出低
            p.state_threshold=cover_st
        vb=model_setting['model_vb']
        # print(("====test model parameters: output_size=%d,time_step=%d,rnn_unit=%d,lstm_layers=%d,model_name=%s,state_window=%d,state_threshold=%e,frame_window=%d,abs_threshold=%d,ab0_threshold=%d,abnormal_threshold=%d")\
        #         % (p.output_size,p.time_step,p.rnn_unit,p.lstm_layers,i,p.state_window,p.state_threshold,p.frame_window,p.abs_threshold,p.ab0_threshold,p.abnormal_threshold))##输出此时正在使用的小模型name
        p.input_size=p.output_size
        
        start_time=time.time()
        tf.reset_default_graph()##清除默认图的堆栈，并设置全局图为默认图 
        p.set_para()##设置权重等结构，之后restore
        result=p.test(data,data_set_name,vb)
        end_time=time.time()
        # print("Test cost:%ss" % (end_time-start_time))
        
        if result==1:##进程正常
            return result
    return result

def main(argv):
    pattern=""
    train_data_arg=""
    test_data_arg=""
    try:
        opts, args = getopt.getopt(argv,"hp:t:o:",\
            ["testdata=","traindata=","outputsize=","timestep=","rnnunit=","lstmlayers=","batchsize=","learningrate=","epoch=","modelname=","statewindow=","framewindow=","absthreshold=","ab0threshold="])
    # except getopt.GetoptError,e:
    #     print(e)
    except getopt.GetoptError:##py3
        # print(e)
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-p"):
            pattern=arg
        elif opt in ("-t"):
            abnormal_threshold=int(arg)
        elif opt in ("-o"):
            cover_st=float(arg)
        elif opt in ("--traindata"):##由集成程序处理完输入后，直接赋值。是一个目录或一个文件
            train_data_arg=arg
        elif opt in ("--testdata"):##测试文件
            test_data_arg=arg
        elif opt in ("--outputsize"):
            output_size=int(arg)
        elif opt in ("--timestep"):
            time_step=int(arg)
        elif opt in ("--rnnunit"):
            rnn_unit=int(arg)
        elif opt in ("--lstmlayers"):
            lstm_layers=int(arg)
        elif opt in ("--batchsize"):
            batch_size=int(arg)
        elif opt in ("--learningrate"):
            lr=float(arg)
        elif opt in ("--epoch"):
            epoch_num=int(arg)
        elif opt in ("--modelname"):
            model_name=arg
        elif opt in ("--statewindow"):
            state_window=int(arg)
        elif opt in ("--framewindow"):
            frame_window=int(arg)
        elif opt in ("--absthreshold"):
            abs_threshold=int(arg)
        elif opt in ("--ab0threshold"):
            ab0_threshold=int(arg)

    ##根据参数确定是训练还是测试
    if pattern=="train":
        path=train_data_arg+"/"
        data_set_list=[]
        if os.path.exists(path)==1:##输入是一个目录
            command="file %s* | awk '{print $1}'" % (path)
            fp=os.popen(command, "r")
            ret=fp.read().split('\n')
            fp.close()
            ret.remove('')
            for i in ret:
                data_set_list.append(i.strip(':'))
        else:
            data_set_list.append(train_data_arg)
        print('')
        print('data_set_list',data_set_list)##用于建立该模型的所有数据
        print(("---%s model parameters:output_size=%d,time_step=%d,rnn_unit=%d,lstm_layers=%d,batch_size=%d,lr=%f,epoch_num=%d,model_name=%s,state_window=%d")\
                % (pattern,output_size,time_step,rnn_unit,lstm_layers,batch_size,lr,epoch_num,model_name,state_window))
        input_size=output_size
        train_data_main(input_size,output_size,time_step,rnn_unit,lstm_layers,batch_size,lr,epoch_num,model_name,state_window,data_set_list)
    elif pattern=="test":##这里的输出会在集成程序中用到，以最后确定哪些是异常，此处可打印别的，但最后一定要打印name或abnormal用于集成程序统计结果
        model_list=[]
        command="file %s/* | awk '{print $1}'" % (model_name)
        fp=os.popen(command,"r")
        ret=fp.read().split('\n')
        fp.close()
        ret.remove('')
        if len(ret)!=0:
            for i in ret:
                model_list.append(i.strip(':'))
        result=test_data_main(model_list,test_data_arg,abnormal_threshold,frame_window,abs_threshold,ab0_threshold,cover_st)
        if result==1:##最终打印结果，用于集成时统计
            print(test_data_arg)
        else:
            print(test_data_arg)
            print('Abnormal')

if __name__ == "__main__":
    main(sys.argv[1:])
