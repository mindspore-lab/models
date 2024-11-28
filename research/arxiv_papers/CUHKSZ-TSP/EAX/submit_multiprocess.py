import time
import os
import sys
import random
import getopt
from multiprocessing import Pool
def run_task(cmd, count):
    print "Task %s (pid=%s) is running..." %(count, os.getpid())
    os.system(cmd)
    print "Task %s end." %count
 
def main(argv):

    num_threads=0
    count=1
    mainpid=''

    try:
        opts, args = getopt.getopt(argv,"c:h",["count="])
    except getopt.GetoptError:
        print 'name.py -c num_threads'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'name.py -c num_threads'
            sys.exit()
        elif opt in ("-c", "--count"):
            num_threads = int(arg)

    mainpid=os.getpid()        
    print "Main process %s" %mainpid
    p = Pool(processes= num_threads)

    with open("subtsk2.sh", 'r') as fr:
        data=fr.readlines()
        for cmd in data:
            p.apply_async(run_task, args= (cmd,count))
            count=count+1
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print "All subprocess done."

if __name__ == "__main__":
    main(sys.argv[1:])
