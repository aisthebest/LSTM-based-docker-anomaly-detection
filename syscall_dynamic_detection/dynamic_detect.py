#!/usr/bin/python
#coding=utf-8
##整体封装程序，利用已经建立好的模型，检测某个运行中容器的进程
##python dynamic_detect.py -c 容器id -t 测试间隔时间/重启另一个采集程序间隔时间 -m 模型名（里面包含很多小模型）
import sys, getopt
import os
import signal
import subprocess
import shutil
import commands
import time

def sigint_handle(signum, frame):
	print("dynamic_detect receive SIGINT")
	if getdata_pid!='':
		os.kill(getdata_pid,signal.SIGINT)##当前退出，则把getdata也退出，相应的strace也会退出
	##清除模型测试时用的保存数据的临时目录：容器id/容器id_1/2/3...
	if contid!='':
		cleanc="file * | grep directory | grep %s | awk '{print $1}'" % (contid)
		ret,output=commands.getstatusoutput(cleanc)##py2.7
		# ret,output=subprocess.getstatusoutput(command)##py3
		if output!='':
			for i in output.split('\n'):
				if os.path.exists(i.strip(':')):
					print('dynamic_detect rmtree this dir',i.strip(':'))
					shutil.rmtree(i.strip(':'))	
	sys.exit(0)

getdata_pid=''
contid=''
tmp=1
def main(argv):
	global getdata_pid,contid,tmp
	signal.signal(signal.SIGINT, sigint_handle)
	contid=''
	collect_time=30##默认检测时间间隔
	model='lstm_model'##默认模型名
	try:
		opts, args = getopt.getopt(argv,"hc:t:m:")
	except getopt.GetoptError:
		print 'usage: dynamic_detect.py -c <container id> -t <collect_time> -m <model name>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'usage: dynamic_detect.py -c <container id> -t <collect_time> -m <model name>'
			sys.exit()
		elif opt in ("-c"):
			contid = arg##此处封装就只考虑一次一个容器
		elif opt in ("-t"):
			collect_time = int(arg)
		elif opt in ("-m"):
			model = arg
	#print 'container list:', contlist
	##当前处于running的容器列表
	lcontlist = []
	command = os.popen("docker ps | awk 'NR==2,NR==0 {print $1,$NF}'", "r")#id name
	for lcontnum, lcont in enumerate(command, 1):#lcont:68b4a7cdaf58 mysql
		lcontlist.append(lcont.split(" ")[0])
		lcontlist.append(lcont.split(" ")[1].strip('\n'))

	if contid in lcontlist: #检测正处于运行状态容器
		collect_data_c="python getdata.py -c %s" % (contid)
		while 1:
			try:
				collect_subpid=subprocess.Popen(collect_data_c, shell = True)
			except:
				print("execute collect data error")
			getdata_pid=collect_subpid.pid
			print('getdata_pid',getdata_pid)
			time.sleep(collect_time)##当前进程停止一定时间，strace还是在采集数据
			os.kill(getdata_pid,signal.SIGINT)##杀掉当前getdata及相应的strace，把当前数据目录改名用于检测，然后再启动新的getdata
			rename="mv %s %s_%s" % (contid,contid,str(tmp))
			ret,output=commands.getstatusoutput(rename)
			if ret!=0:
				print("dynamic_detect rename data dir error")
				sys.exit(2)
			##对临时目录数据进行测试，新建子进程，测试完就删除此临时目录
			test_data_c="python lstm_wrapper.py -p test --modelname %s --data %s_%s" % (model,contid,str(tmp))##重要：modelname需要确定好
			try:
				test_subpid=subprocess.Popen(test_data_c, shell = True)
			except:
				print("execute test error")
			print('')
			print("dynamic_detect time's up, start test")
			print("test_pid",test_subpid.pid)
			tmp+=1
	else:
		print '------ERROR : The container : %s is not open' % (contid)
		sys.exit(2)

if __name__ == "__main__":
	main(sys.argv[1:])
