/*
* Name        : Parameter.cpp
* Author      : Sipeng, Sun
* Description : This file implement all functions in Parameter.h
*/

#include <iostream>
#include <string>

#include "Parameter.h"


std::string INPUT_FILE_PATH;		
std::string OUTPUT_FILE_PATH;

int SEED = 1;		
int TIME = -1;		
int SOLUTION_NUM = 10;		
int OPERATION_NUM = 90;		
int REMOVE_EDGE_NUM = 500;


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
void ReadParameter(int argc, char* argv[]) {
	while (argc != 1) {
		if (std::string(argv[argc - 2]) == "--seed") SEED = atoi(argv[argc - 1]);
		else if (std::string(argv[argc - 2]) == "-input") INPUT_FILE_PATH = std::string(argv[argc - 1]);
		else if (std::string(argv[argc - 2]) == "-output")  OUTPUT_FILE_PATH = std::string(argv[argc - 1]);
		else if (std::string(argv[argc - 2]) == "-time")  TIME = atoi(argv[argc - 1]);
		else if (std::string(argv[argc - 2]) == "-sol")  SOLUTION_NUM = atoi(argv[argc - 1]);
		else if (std::string(argv[argc - 2]) == "-ope")  OPERATION_NUM = atoi(argv[argc - 1]);
		else if (std::string(argv[argc - 2]) == "-edge")  REMOVE_EDGE_NUM = atoi(argv[argc - 1]);
		else {
			std::cout << "ERROR" << std::endl
				<< "Unknown parameter!" << std::endl;
			exit(EXIT_FAILURE);
		}
		argc = argc - 2;
	}
}
/// <summary>
/// output all the parameters to help the user check 
/// whether the parameters are all in the expectations.
/// </summary>
void OutputParameter() {
	std::cout << "INPUT_FILE_PATH : " << INPUT_FILE_PATH << std::endl
		<< "OUTPUT_FILE_PATH : " << OUTPUT_FILE_PATH << std::endl
		<< "SEED : " << SEED << std::endl
		<< "TIME : " << TIME << std::endl
		<< "SOLUTION_NUM : " << SOLUTION_NUM << std::endl
		<< "OPERATION_NUM : " << OPERATION_NUM << std::endl
		<< "REMOVE_EDGE_NUM : " << REMOVE_EDGE_NUM << std::endl;
	return;
}


