
Function of each forder and file:

***************************************************************************

main.cpp :

It contains "main" function and calls many functions from different files to solve a TSP problem. 
The whole process how the program solves the problem is in this file.

***************************************************************************

Solve_subTSP folder :

It stores the code of improved EAX which meets our requirement.

The function "Solve_Sub_TSP" in Solve_Sub_TSP.cpp is the main function we call in main.cpp to solve the sub-tsp problem.

***************************************************************************

Route.cpp && Route.h :

It contains some basic functions about a solution like writing a tour, and structs.

The function "PRE_POS" and "NEXT_POS" can obtain the previous and the next postion in a circle solution.

The function "ReadProblem" can read a TSP problem and obtain some important information.

The function "WriteTour" can output a tour.

The function "DIS" can compute the distance between two cities.

The function "LENGTH" can compute the length of the tour.

The function "IsMustConnect" can judge whether two cities are connected or not.

***************************************************************************

Parameter.cpp && Parameter.h :

It deals with the paramaters users input in cmd. 

The function "ReadParameter" can read the parameter information to initialize our program.
The function "OutputParameter" can output the parameter information so that users can check whether they are in expection.

***************************************************************************

TwoOpt.cpp && TwoOpt.h :

Its code is about two-opt method, whose goal is to solve a sub-tsp problem when initializing a solution.

The function "TwoOptOptimise" can solve a sub-tsp problem, and it is frequently called in InitRoute.cpp

***************************************************************************

InitRoute.cpp && InitRoute.h :

It can be used to form a initial feasible solution.

The function "InitPath" can initialize a solution, and it is mainly called in main.cpp.

***************************************************************************

SmallerTSP.cpp && SmallerTSP.h :

It is about how to compress a solution, solve the compressed problem and uncompress the solution.

The function "ZipTSP" can compress a solution.

The function "RemoveEdge" can remove an amount of edges.

The function "GetDisMatrix" can get the dismatrix of the compressed solution.

The function "UnzipTSP" can uncompress the solution.
