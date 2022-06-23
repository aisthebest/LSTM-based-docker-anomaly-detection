#!/usr/bin/python
#coding=utf-8
##采集正在运行中的容器内进程系统调用数据
##python getdata.py -c 容器id
import sys, getopt
import os
import signal
import subprocess
import shutil
import commands

subpids = []
def tracepid(contid, pidlist):
	formatpid = ','.join(pidlist)
	if os.path.exists(contid)==1:
		shutil.rmtree(contid)
		os.mkdir(contid)
	else:
		os.mkdir(contid)
	file_path="./%s/%s"%(contid,contid)
	tracec = "strace -e trace=all -f -F -ff -o %s -p %s -n %s" % (file_path, formatpid, pidlist[0])
	#print tracec
	subpids.append(subprocess.Popen(tracec, shell = True))##启动strace采集数据
	print '++++++strace:%d trace %s' % (subpids[-1].pid, contid)
	return

def cid2pid(contid):
	pidlist = []
	print '++++++container : %s' % (contid)
	
	ppidc = "docker top %s | awk 'NR==2 {print $3}'" % (contid)
	pptop = os.popen(ppidc, "r")
	for ppid in pptop:
		pidlist.append(ppid.strip('\n'))

	pidc = "docker top %s | awk 'NR==2,NR==0 {print $2}'" % (contid)
	ptop = os.popen(pidc, "r")
	for pnum, pid in enumerate(ptop, 1):
		pidlist.append(pid.strip('\n'))
	print 'pidlist:', pidlist 
	return pidlist

def sigint_handle(signum, frame):
	print('getdata receive SIGINT')
	sys.exit(0)

contid=''
pidlist=[]
def main(argv):
	global contid,pidlist
	signal.signal(signal.SIGINT, sigint_handle)
	contlist = ''
	try:
		opts, args = getopt.getopt(argv,"hc:")
	except getopt.GetoptError:
		print 'usage: getdata.py -c <container id>'
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print 'usage: getdata.py -c <container id>'
			sys.exit()
		elif opt in ("-c"):
			contlist = arg
	#print 'container list:', contlist
   
	for contnum, contid in enumerate(contlist.split(","), 1):
		lcontlist = []
		command = os.popen("docker ps | awk 'NR==2,NR==0 {print $1,$NF}'", "r")#id name
		for lcontnum, lcont in enumerate(command, 1):
			lcontlist.append(lcont.split(" ")[0])
			lcontlist.append(lcont.split(" ")[1].strip('\n'))

		if contid in lcontlist:##对处于运行状态容器采集数据
			pidlist=cid2pid(contid)
			tracepid(contid, pidlist)
		else:
			print '------ERROR : The container : %s is not open' % (contid)
			continue

	for subpid in subpids:#等待所有strace，这样getdata退出，strace也退出
		subpid.wait()
		print '++++++getdata wait for strace:%d exit with %s' % (subpid.pid, subpid.returncode)##如果sigint是return，ctrl+c会显示这里。改为直接退出，strace也会退出
		

if __name__ == "__main__":
	main(sys.argv[1:])