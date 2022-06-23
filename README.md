# LSTM-based-docker-anomaly-detection
## 作者
    王玉龙 中国工程物理研究院计算机应用研究所/四川大学网络空间安全研究院
    email：wangyulong@caep.cn
    
## 1、文件说明
    
    strace是数据采集程序
    dynamic_detect.py是集成程序
    getdata.py是结合容器的采集封装程序
    lstm_main.py是建模和检测的主体程序
    lstm_wrapper.py是建模和检测的封装程序

## 2、使用

2.1 数据采集

命令：
  首先需要进入strace目录
  
        make && make install
        python getdata.py –c 容器id
    
效果：
    在当前目录下新建一个以容器id命名的文件夹，里面存储了该容器内运行的所有进程和线程产生的系统调用数据
    
2.2 数据建模

命令：

        python lstm_wrapper.py -p train --data traindata --modelname lstm_model --outputsize 314 --timestep 15 --epoch 100 --rnnunit 100 --lstmlayers 2 --learningrate 0.05 --statewindow 5
        
  也可采用默认模型参数设置，直接：
        python lstm_wrapper.py -p train --data traindata --modelname lstm_model
        
  参数含义如下：

![Fig 1](https://github.com/aisthebest/LSTM-based-docker-anomaly-detection/blob/main/Fig1.jpg)

效果：

  对traindata目录下的所有文件和文件夹各自建立小模型，统一存放到lstm_model这个目录下，
  lstm_model目录作为大模型名。并产生一个名为lstm_model_train_loss的目录，里面存放每个小模型的loss图，可由此在一定程度上判定是否收敛。（可在lstm_main.py中注释此图的生成）

 2.3 异常检测
 
 命令：
      
        python dynamic_detect.py -c 容器id –m 模型名称（该文件夹内包含很多小模型）-t 测试时间间隔
        
   ![Fig 2](https://github.com/aisthebest/LSTM-based-docker-anomaly-detection/blob/main/Fig2.jpg)

可以看到追踪期间，容器内进程的新建，以及bash进程装载别的程序后会将相应的存储文件进行修改，以便后续异常日志准确输出进程信息（这些输出信息可在strace目录下的strace.c程序中注释掉）。

设定时间到了后，会把原strace给kill，然后把数据所在目录改名为临时目录，用于检测，然后新起采集程序，此时的采集和检测互不干扰，会将检测结果输出。

   ![Fig 3](https://github.com/aisthebest/LSTM-based-docker-anomaly-detection/blob/main/Fig3.jpg)

 效果：
      当前目录下会产生一个detect_result的异常日志文件，内容如下：
   ![Fig 4](https://github.com/aisthebest/LSTM-based-docker-anomaly-detection/blob/main/Fig4.jpg)
   
  检测过程中会有用于测试的临时目录，在测试完毕即会删除。用ctrl+c中止该程序时，也会将所有临时目录删除。
  
## 注意

1、关于检测时的默认参数设置在lstm_wrapper.py中，可根据需要自行调节
   ![Fig 5](https://github.com/aisthebest/LSTM-based-docker-anomaly-detection/blob/main/Fig5.jpg)
   
2、进程池大小也在lstm_wrapper.py中

