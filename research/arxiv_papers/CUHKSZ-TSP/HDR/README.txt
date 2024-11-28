***************************************************************************

Functions of each folder:

Src	: store the code

***************************************************************************

Functions of each file:

compile.sh: compile the code and generate the execution file "SOLVER"

****************************************************************************

Execution steps:


Step 1: compile the code and generate the execution file "SOLVER"

./compile.sh

Step 2: run the instance you want

./SOLVER.sh -input A -output B --seed C -time D -sol E -ope F -edge G

Parameter "-input" is mandatory. A should be the path of the problem file which is in tsplib format.

Parameter "-output" is mandatory. B should be the path of output file where the best tour is to be written.

Parameter "--seed" is optional. It controls the initial seed. The default value is 1.

Parameter "-time" is optional. It controls the total run time. If it isn't specified, the program doesn't terminate because of time.

Parameter "-sol" is optional. It controls the number of solutions generated. The default value is 10.

Parameter "-ope" is optional. It controls the number of destroy & repair operations in each hierarchy. The number of destroy & repair operations in each hierarchy is n/F (n is the number of cities in each hierarchy). The default value is 90.

parameter "-edge" is oprional. It controls the number of destoryed edge during each destory&repair operation. The default value is 500.


*******************************************************************************

Our experiment 1 and 2 in the paper:

For each instance, we run 10 times with C from 1 to 10. Set D n/5(n is the size of the problem) if n is less than 1,000,000, otherwise set D 100,000.
Other parameters all adopt the default value. 

Therefore, the cmd is:
./SOLVER -input A -output B --seed C(1~10) -time n/5
or:
./SOLVER -input A -output B --seed C(1~10) -time 100,000


