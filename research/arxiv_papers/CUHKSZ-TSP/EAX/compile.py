import os
import sys
import shutil
import getopt

##delete and rebuild the output
shutil.rmtree('output')
os.mkdir('output')
##delete and rebuild the error
shutil.rmtree('error')
os.mkdir('error')
##delete EDGF
shutil.rmtree('Sol')
os.mkdir('Sol')

os.system("rm output_all_task.log")
os.system("rm mem0_mtrpp.sh")
os.system("rm subtsk*")
os.system("rm output_all*")


def main(argv):
    time=""
    memory=""
    cpu=0
    lim=0

    try:
        opts, args = getopt.getopt(argv,"l:t:hc:",["time=","limi=", "cpu="])
    except getopt.GetoptError:
        print 'name.py -t time -l execution_times'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'name.py -t time -l execution_times'
            sys.exit()
        elif opt in ("-t", "--time"):
            time = arg
        elif opt in ("-l", "--limi"):
            lim = int(arg)
        elif opt in ("-c", "--cpu"):
            ccc = int(arg)

    cppname="exe_cpp/*.cpp exe_cpp/Object/*.cpp exe_cpp/NewEAX/*.cpp"
    exename='MTRPP_S'

    cmd1='g++ %s -O3 -lm -Wall -o %s'%(cppname,exename)
    os.system(cmd1)

    count=0

    ct=0
    for f in os.listdir('Ins'):
        with open("subtsk2.sh",'a') as fw:
            for j in range(lim):
                fw.write('./%s --seed %d -rep %d -i %s -s %s -d %s 1>output/output_%s_%d.out 2>error/error_%s_%d.err\n'%(exename, j, j, f, "Ins" ,"Sol", f, j, f,j))
                count=count+1

    print count

    os.popen("chmod u+x subtsk*.sh")


if __name__ == "__main__":
    main(sys.argv[1:])
