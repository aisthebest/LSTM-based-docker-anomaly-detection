#coding=utf-8
##建模和检测的封装程序，实现多进程和默认参数
##建模时一个目录里多个文件夹或者文件，建立多个小模型，保存到modelname文件夹中
##检测时一个目录都是待检测文件，modelname里包含很多小模型文件夹
##python lstm_wrapper.py -p train --data traindata --modelname lstm_model --outputsize 314 --timestep 15 --epoch 100 --rnnunit 100 --lstmlayers 2 --learningrate 0.05 --statewindow 5 
##python lstm_wrapper.py -p test --data testdata --modelname lstm_model --framewindow 20 --absthreshold 2 --ab0threshold 5 -t 2 -o 1.0e-7
import sys
import getopt
import time
import os
import multiprocessing
import commands
import subprocess
import pandas as pd
import numpy as np
import signal
import shutil

def call_train(msg):
    print('train pid:',os.getpid())
    command="python lstm_main.py -p train --traindata %s --outputsize %d --timestep %d --rnnunit %d --lstmlayers %d --batchsize %d --learningrate %f --epoch %d --modelname %s --statewindow %d" \
    % (msg[0],msg[1],msg[2],msg[3],msg[4],msg[5],msg[6],msg[7],msg[8],msg[9])
    print("command",command)
    subpid=subprocess.Popen(command, shell = True)
    subpid.wait()

def call_test(msg):##注意测试阶段的模型必须放在一个文件夹内，包含很多模型文件夹
    print('test subpid:%ld for %s' % (os.getpid(),msg[1]))
    command="python lstm_main.py -p test --modelname %s --testdata %s -t %d --framewindow %d --absthreshold %d --ab0threshold %d -o %e" \
    % (msg[0],msg[1],msg[2],msg[3],msg[4],msg[5],msg[6])
    # print("command",command)
    ret,output=commands.getstatusoutput(command)##py2.7
    # ret,output=subprocess.getstatusoutput(command)
    # print(output)##输出信息
    return output.split('\n')[-1]##如果是normal则返回对应name，如果是异常则返回Abnormal，用于后续整体输出到日志

##用法说明
def usage():
    print("---usage for train:")
    print("python lstm_main.py -p train --data <data_set> --outputsize 1 --timestep 15 --rnnunit 100 --lstmlayers 2 --batchsize 100 --learningrate 0.005 --epoch 100 --modelname <model> --statewindow 5")
    print("---usage for test:")
    print("python lstm_main.py -p test --data <data_set> --modelname <model> -t <abnormal_threshold> --framewindow 20 --absthreshold 2 --ab0threshold 5 -o 1.0e-7")

def main(argv):
    ##默认参数
    output_size=set_outputsize()  ##输出概率向量长度，为系统调用类型个。由strace确定，可由参数覆盖指定
    input_size=1  ##后会被output_size覆盖
    rnn_unit=100  ##隐层神经元的个数。如果数据量大，行为迥异，可增大此值，实验800个不同进程时，200最优
    lstm_layers=2  ##隐层层数。数据量大，5最优
    epoch_num=100  ##所有样本的迭代次数。数据量大，200最优
    time_step=15  ##时间步长，表示模型一次接收的短序列长度，用于预测下一个系统调用
    batch_size=100  ##更新一次网络参数使用的样本数量
    lr=0.005  ##学习率
    model_name='lstm_model'  ##默认形成的大模型名
    state_window=5  ##一个状态的系统调用数量，建模时测试训练数据确定状态的最小出现概率作为阈值，检测时衡量每个状态的出现概率

    state_threshold=1  ##状态异常阈值，由训练阶段确定，测试阶段使用
    frame_window=20  ##局部帧大小
    abs_threshold=2  ##检测方法1中，局部帧内状态异常阈值数，即一帧内有阈值个状态异常，则此帧异常
    ab0_threshold=5  ##检测方法2中，局部帧内异常0阈值数，即帧异常
    abnormal_threshold=3  ##异常局部帧有阈值个，则判定进程异常

    cover_st=0  ##若建模时没有收敛，模型自定义的状态异常阈值太低，则可由此自定义状态阈值，覆盖模型设定的值。默认为0不覆盖
    
    ##提取命令行参数，确定训练模型参数，或测试的异常标准
    pattern=""
    data=""
    try:
        opts, args = getopt.getopt(argv,"hp:t:o:",\
            ["data=","outputsize=","timestep=","rnnunit=","lstmlayers=","batchsize=","learningrate=","epoch=","modelname=","statewindow=","framewindow=","absthreshold=","ab0threshold="])
    # except getopt.GetoptError,e:##py2
    #     print(e)
    except getopt.GetoptError:##py3
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-p"):##建模还是测试
            pattern=arg
        elif opt in ("-t"):
            abnormal_threshold=int(arg)
        elif opt in ("-o"):
            cover_st=float(arg)
        elif opt in ("--data"):##输入一个目录，里面文件和文件夹都单独建模（文件夹可能是一个进程多次运行）。输入一个目录用于测试
            data=arg
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

    ##由参数确定训练或测试的数据集
    if pattern=="":
        print("Please input pattern, train or test")
        usage()
        sys.exit(2)
    if data=="":
        print("Please input data_set")
        usage()
        sys.exit(2)
    data_set_list=[]    
    command="file %s/* | awk '{print $1}'" % (data)
    fp=os.popen(command,"r")
    ret=fp.read().split('\n')
    fp.close()
    ret.remove('')
    if len(ret)!=0:
        for i in ret:
            data_set_list.append(i.strip(':'))
    else:
        print("Please input data")
    # print('data_set_list',data_set_list,len(data_set_list))

    if pattern=="train":
        start_time=time.time()
        pool=multiprocessing.Pool(processes=1)
        for data_set_name in data_set_list:##data_set_name是一个文件或者一个目录，最后形成一个模型
            pool.apply_async(call_train, ([data_set_name,output_size,time_step,rnn_unit,lstm_layers,batch_size,lr,epoch_num,model_name,state_window], ))
        pool.close()
        pool.join()
        end_time=time.time()
        print('')
        print("All train costs:%ss" % (end_time-start_time))##建立所有模型耗时
    elif pattern=="test":
        if os.path.exists(model_name)!=1:
            print("Model:%s doesn't exists! Please input right model_name!" % (model_name))
            sys.exit(2)
        non_detect=0
        pool=multiprocessing.Pool(processes=10)
        result=[]
        for data_index, data_set_name in enumerate(data_set_list):
            ##首先看测试数据量是否够检测
            command="du -h %s | awk '{print $1}'" % (data_set_name)
            fp=os.popen(command,"r")
            ret=fp.read().strip('\n')
            fp.close()
            if ret=='0':##该数据文件没内容，大小为0k，如果不处理就read_csv，会报错pd
                # print("The number of system calls in this file is not enough for testing : %s" %  (data_set_name))##最后可以直接判定正常，只输出异常即可
                non_detect+=1
                data_set_list[data_index]='non_detect'##之后的小模型就不检测该数据文件了
                continue
            df=pd.read_csv(data_set_name,header=None)
            data_c=np.array(df)
            len_of_data=len(data_c)
            if(len_of_data < time_step+1):##若文件内不够一个序列，则跳过该文件
                # print("The number of system calls in this file is not enough for testing : %s" %  (data_set_name))##最后可以直接判定正常，只输出异常即可
                non_detect+=1
                data_set_list[data_index]='non_detect'##之后的小模型就不检测该数据文件了
                continue
            ##可检测则发起进程检测
            result.append(pool.apply_async(call_test, ([model_name,data_set_name,abnormal_threshold,frame_window,abs_threshold,ab0_threshold,cover_st], )))
        pool.close()
        pool.join()
        testre=[]
        for i in result:
            testre.append(i.get())##如果是Abnormal即异常，若不是则为相应name为正常
        abnormal_list=[]
        for i in data_set_list:
            if i not in testre and i!='non_detect':##提取出异常进程信息
                abnormal_list.append(i)
        print('Abnormal_list',abnormal_list,len(abnormal_list))
        
        ##输出到检测日志
        result=open("./detect_result",'a')
        if len(abnormal_list)!=0:##结果是：['1bdfe27a386d_1/1bdfe27a386d.bash.6403', '1bdfe27a386d_1/1bdfe27a386d.ls.27326']
            for i in abnormal_list:
                print >> result,'process name:%s,pid:%s in container %s is malicious' %(i.split('.')[1],i.split('.')[2],i.split('.')[0].split('/')[-1])
        result.close()

        # ##检测完毕删除该存储数据的临时目录
        if os.path.exists(data)==1:
            shutil.rmtree(data)
        
def set_outputsize():
    fp=os.popen("strace -N n","r")
    output_size=int(fp.read().strip('\n'))
    fp.close()
    return output_size

def sigint_handle(signum, frame):
    os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handle)
    main(sys.argv[1:])
