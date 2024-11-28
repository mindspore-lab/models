***************************************************************************

Functions of each folder:

DOC : some papers about LKH

SRC	: store the code

***************************************************************************

Functions of each file:

C10k.0_1.par: demo .par file

Makefile: compile the code and generate the execution file "LKH"

****************************************************************************

Execution steps:


Step 1: compile the code and generate the execution file "SOLVER"

make

step 2: write .par file 

The followinng fomat is only for our experiment.  
"
PROBLEM_FILE = A
CANDIDATE_SET_TYPE = POPMUSIC
INITIAL_PERIOD = 1000
TIME_LIMIT = B
RUNS = 1
SEED = C
TOUR_FILE = D
"

A is the path of the problem file

B controls the total run time

C controls the initial seed

D controls the path of the file where the best tour is to be written


Step 2: run the instance you want

./LKH xxx.par

xxx.par is the .par file you want to run

*******************************************************************************

Our experiment :

For each instance, we run 10 times with SEED from 1 to 10. Set TIME n/5(n is the size of the problem) if n is less than 1,000,000, otherwise set TIME 100,000.
Other parameters is like the demo shows. 


