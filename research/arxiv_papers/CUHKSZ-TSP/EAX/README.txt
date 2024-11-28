***************************************************************************

Functions of each folder:

Detail 	: store the final results of each execution
error  	: store the error file of each execution
exe_cpp	: code of EAX
Ins		: Instances of tested, the benchmark instances are too large, please download them from the TSP websites and put them here. For the file format, see E10K.0.
output	: store the part results of each execution
Sol		: store the part results of each execution
summary_new: the summary of results

***************************************************************************

Functions of each file:

compile.py: compile the code and generate the list of tasks, you could change some related settings in the file 'LaRen.sh'

fn2.txt: the list of instances, you could change it as you like

LaRen.sh: compile file, the option "-l 10" means the number of executions for each file.

MTRPP_S: execution file of EAX.

Qsub.sh: submitting script, the option "-c 10" means the number of threads to run the tasks.

submit_multiprocess.py: submitting script, you could change some related settings in the file 'Qsub.sh'.

subtsk2.sh: list of executions

syngen_new.py: the script to generate the summary file


****************************************************************************

Execution steps:


Step 1: compile the code and generate the list of tasks

./LaRen.sh

Step 2: run the tasks and waiting

./Qsub.sh


Step 3: when the executions are finished, summarize the results, and you could see the final summary file in 'summary_new'
The Columns in the file represent instance, best-known, best, average, preprocessing-time, all-time, the gap between best and best-knowon, and the gap between average and best-known, respectively.

python2 syngen_new.py
