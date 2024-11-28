/*
* Name        : Parameter.h
* Author      : Sipeng, Sun
* Description : This file contails all the parameters that 
*				the user can control.
*/


#ifndef _Parameter_h
#define _Parameter_h

#include <string>

extern std::string INPUT_FILE_PATH;		// the path of problem file in tsplib
extern std::string OUTPUT_FILE_PATH;		// the path of output file where stores the best solution

extern int SEED;		// the randon seed
extern int TIME;		// the total time the program runs
extern int SOLUTION_NUM;		// the number of solution each hierarchy
extern int OPERATION_NUM;		// the number of operation for each solution is n/OPERATION_NUM each hierarchy, where n is the size of cities of each hierarchy
extern int REMOVE_EDGE_NUM;	// the number of edge to delete each destroy&repair operation 

/// <summary>
/// read the parameter from cmd
/// </summary>
/// 
/// <param name="argc">
///	int type, 
/// indicates the size of char array argv
/// </param>
/// 
/// <param name="argv">
///	char array type,
/// which is the cmd
/// </param>
void ReadParameter(int argc, char* argv[]);
/// <summary>
/// output all the parameters to help the user check 
/// whether the parameters are all in the expectations.
/// </summary>
void OutputParameter();


#endif // !_Parameter_h