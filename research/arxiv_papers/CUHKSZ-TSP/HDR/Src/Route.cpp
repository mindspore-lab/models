/*
* Name        : Route.cpp
* Author      : Sipeng, Sun
* Description : This file implements the functions in Route.h
*/

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "Route.h"

int TSP_SIZE = 0;
std::string INSTANCE = "";
std::string INSTANCE_TYPE = "";


/// <summary>
/// obtain the previous position
/// </summary>
/// 
/// <param name="now_pos">
/// int type,
/// shows the current position
/// </param>
/// 
/// <param name="size">
/// int type,
/// indicates the size of array
/// </param>
/// 
/// <param name="move_step">
/// int type,
/// indicates the number of steps to move forward
/// </param>
/// 
/// <returns>
/// int type,
/// the position after moving
/// </returns>
int PRE_POS(int now_pos, int size, int move_step) {
	return (now_pos - move_step + size) % size;
}
/// <summary>
/// obtain the next position
/// </summary>
/// 
/// <param name="now_pos">
/// int type,
/// shows the current position
/// </param>
/// 
/// <param name="size">
/// int type,
/// indicates the size of array
/// </param>
/// 
/// <param name="move_step">
/// int type,
/// indicates the number of steps to move backward
/// </param>
/// 
/// <returns>
/// int type,
/// the position after moving
/// </returns>
int NEXT_POS(int now_pos, int size, int move_step) {
	return (now_pos + move_step) % size;
}

/// <summary>
/// read information about the problem from the input file
/// </summary>
/// 
/// <param name="points">
/// Point array, 
/// stores the information about the cordinates of cities 
/// </param>
void ReadProblem(Point*& points) {
	std::ifstream ifile(INPUT_FILE_PATH.c_str());
	if (ifile.fail()) {
		std::cout << "ERROR" << std::endl
			<< "FAIL TO OPEN THE INPUT FILE" << std::endl
			<< INPUT_FILE_PATH << std::endl;
		exit(EXIT_FAILURE);
	}

	std::string line;
	while (getline(ifile, line)) {
		if (line.substr(0, 4) == "NAME") {
			int pos = line.find(":");
			INSTANCE = line.substr(pos + 2);
		}
		else if (line.substr(0, 9) == "DIMENSION") {
			int pos = line.find(":");
			TSP_SIZE = atoi(line.substr(pos + 2).c_str());
		}
		else if (line.substr(0, 16) == "EDGE_WEIGHT_TYPE") {
			int pos = line.find(":");
			INSTANCE_TYPE = line.substr(pos + 2);
		}
		else if (line.substr(0, 18) == "NODE_COORD_SECTION") break;
	}

	points = new Point[TSP_SIZE];
	int temp;
	for (int i = 0; i < TSP_SIZE; i++) {
		ifile >> temp;
		ifile >> points[i].x;
		ifile >> points[i].y;
	}
	ifile.close();
}
/// <summary>
/// output the tour to the output file.
/// </summary>
/// 
/// <param name="route">
/// int array,
/// the route which is going to output
/// </param>
/// 
/// <param name="Opt">
/// Type_Route_Length type,
/// the length of the output tour.
/// </param>
void WriteTour(int* route, Type_Route_Length& Opt) {
	std::ofstream ofile(OUTPUT_FILE_PATH.c_str());
	if (ofile.fail()) {
		std::cout << "ERROR" << std::endl
			<< "FAIL TO OPEN THE OUTPUT FILE" << std::endl
			<< OUTPUT_FILE_PATH << std::endl;
		exit(EXIT_FAILURE);
	}

	ofile << "NAME : " << INSTANCE << "." << Opt << ".tour" << std::endl;
	ofile << "COMMENT : Length = " << Opt << std::endl;
	ofile << "TYPE : TOUR" << std::endl;
	ofile << "DIMENSION : " << TSP_SIZE << std::endl;
	ofile << "TOUR_SECTION" << std::endl;

	int pos = 0;
	for (pos; pos < TSP_SIZE && route[pos]; pos++);
	for (int i = 0; i < TSP_SIZE; i++) {
		ofile << route[pos] + 1 << std::endl;
		pos = NEXT_POS(pos, TSP_SIZE);
	}
	ofile << -1 << std::endl
		<< "EOF" << std::endl;

	ofile.close();
}


/// <summary>
/// compute the distance between two cities if tsp is EUC_2D type
/// </summary>
/// 
/// <param name="point1">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <param name="point2">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <returns>
/// int type,
/// whichs is the distance detween the two cities.
/// </returns>
int EUC_2D(Point point1, Point point2) {
	double xd = point1.x - point2.x;
	double yd = point1.y - point2.y;
	return (int)round(sqrt(xd * xd + yd * yd));
}
/// <summary>
/// compute the distance between two cities if tsp is MAN_2D type
/// </summary>
/// 
/// <param name="point1">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <param name="point2">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <returns>
/// int type,
/// whichs is the distance detween the two cities.
/// </returns>
int MAN_2D(Point point1, Point point2) {
	double xd = abs(point1.x - point2.x);
	double yd = abs(point1.y - point2.y);
	return (int)round(xd + yd);
}
/// <summary>
/// compute the distance between two cities if tsp is MAX_2D type
/// </summary>
/// 
/// <param name="point1">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <param name="point2">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <returns>
/// int type,
/// whichs is the distance detween the two cities.
/// </returns>
int MAX_2D(Point point1, Point point2) {
	double xd = abs(point1.x - point2.x);
	double yd = abs(point1.y - point2.y);
	return (int)std::max(round(xd), round(yd));
}
/// <summary>
/// compute the distance between two cities if tsp is GEO type
/// </summary>
/// 
/// <param name="point1">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <param name="point2">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <returns>
/// int type,
/// whichs is the distance detween the two cities.
/// </returns>
int GEO(Point point1, Point point2) {
	const double pi = 3.141592;
	const double RRR = 6378.388;

	int deg;
	double m;
	double lati, latj, longi, longj;
	double q1, q2, q3, q4, q5;

	deg = int(point1.x);
	m = point1.x - deg;
	lati = pi * (deg + 5.0 * m / 3.0) / 180.0;
	deg = int(point1.y);
	m = point1.y - deg;
	longi = pi * (deg + 5.0 * m / 3.0) / 180.0;

	deg = int(point2.x);
	m = point2.x - deg;
	latj = pi * (deg + 5.0 * m / 3.0) / 180.0;
	deg = int(point2.y);
	m = point2.y - deg;
	longj = pi * (deg + 5.0 * m / 3.0) / 180.0;

	q1 = cos(longi - longj);
	q2 = cos(lati - latj);
	q3 = cos(lati + latj);

	return (int)(RRR * acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0);
}
/// <summary>
/// compute the distance between two cities if tsp is ATT type
/// </summary>
/// 
/// <param name="point1">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <param name="point2">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <returns>
/// int type,
/// whichs is the distance detween the two cities.
/// </returns>
int ATT(Point point1, Point point2) {
	double xd = point1.x - point2.x;
	double yd = point1.y - point2.y;
	double r = sqrt((xd * xd + yd * yd) / 10.0);
	int t = round(r);

	if (t < r) return (t + 1);
	else return t;
}
/// <summary>
/// compute the distance between two cities if tsp is CEIL_2D type
/// </summary>
/// 
/// <param name="point1">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <param name="point2">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <returns>
/// int type,
/// whichs is the distance detween the two cities.
/// </returns>
int CEIL_2D(Point point1, Point point2) {
	double xd = point1.x - point2.x;
	double yd = point1.y - point2.y;
	return (int)ceil(sqrt(xd * xd + yd * yd));
}
/// <summary>
/// compute the distance between two cities if tsp is world tsp
/// </summary>
/// 
/// <param name="point1">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <param name="point2">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <returns>
/// int type,
/// whichs is the distance detween the two cities.
/// </returns>
int WORLD(Point point1, Point point2) {
	double lati, latj, longi, longj;
	double q1, q2, q3, q4, q5;

	lati = point1.x * PI / 180.0;
	latj = point2.x * PI / 180.0;

	longi = point1.y * PI / 180.0;
	longj = point2.y * PI / 180.0;

	q1 = cos(latj) * sin(longi - longj);
	q3 = sin((longi - longj) / 2.0);
	q4 = cos((longi - longj) / 2.0);
	q2 = sin(lati + latj) * q3 * q3 - sin(lati - latj) * q4 * q4;
	q5 = cos(lati - latj) * q4 * q4 - cos(lati + latj) * q3 * q3;
	return (int)(6378388.0 * atan2(sqrt(q1 * q1 + q2 * q2), q5) + 1.0);
}
/// <summary>
/// compute the distance between two cities according to the instance type
/// </summary>
/// 
/// <param name="point1">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <param name="point2">
/// Point type,
/// one of the two cities
/// </param>
/// 
/// <returns>
/// int type,
/// whichs is the distance detween the two cities.
/// </returns>
int DIS(Point point1, Point point2) {
	if (INSTANCE_TYPE == "EUC_2D") return EUC_2D(point1, point2);
	else if (INSTANCE_TYPE == "MAN_2D")	return MAN_2D(point1, point2);
	else if (INSTANCE_TYPE == "MAX_2D")	return MAX_2D(point1, point2);
	else if (INSTANCE_TYPE == "GEO")	return GEO(point1, point2);
	else if (INSTANCE_TYPE == "ATT")	return ATT(point1, point2);
	else if (INSTANCE_TYPE == "CEIL_2D")	return CEIL_2D(point1, point2);
	else if (INSTANCE_TYPE == "MAX_2D")	return MAX_2D(point1, point2);
	else if (INSTANCE_TYPE == "WORLD")	return WORLD(point1, point2);
	else {
		std::cout << "The program doesn't support "
			<< INSTANCE_TYPE << " type!" << std::endl;
		exit(EXIT_FAILURE);
	}
}

/// <summary>
/// compute the length of the tour according to points information
/// </summary>
/// 
/// <param name="route">
/// int array,
/// which is the target route
/// </param>
/// 
/// <param name="points">
/// Point array,
/// which stores the information about cities
/// </param>
/// 
/// <param name="n_element">
/// int type,
/// wihich is the size of the tour
/// </param>
/// 
/// <returns>
/// Type_Route_Length type,
/// which is the length of the target tour
/// </returns>
Type_Route_Length LENGTH(int* route, Point* points, int n_element) {
	Type_Route_Length length = 0;
	for (int i = 0; i < n_element; i++)
		length += DIS(points[route[i]], points[route[NEXT_POS(i, n_element)]]);
	return length;
}
/// <summary>
/// compute the length of the tour according to distance matrix
/// </summary>
/// 
/// <param name="route">
/// int array,
/// which is the target route
/// </param>
/// 
/// <param name="DisMatrix">
/// two dimensional int array,
/// which stores the information about distance
/// </param>
/// 
/// <param name="n_element">
/// int type,
/// wihich is the size of the tour
/// </param>
/// 
/// <returns>
/// Type_Route_Length type,
/// which is the length of the target tour
/// </returns>
Type_Route_Length LENGTH(int* route, int** DisMatrix, int n_element) {
	Type_Route_Length length = 0;
	for (int i = 0; i < n_element; i++)
		length += DisMatrix[route[i]][route[NEXT_POS(i, n_element)]];
	return length;
}

/// <summary>
/// judge whether two cities must be connected
/// </summary>
/// 
/// <param name="node1">
/// int type,
/// one of the two cities
/// </param>
/// 
/// <param name="node2">
/// int type,
/// one of the two cities
/// </param>
/// 
/// <param name="nodes">
/// vector of New_Node,
/// which stores the information of cities after compression
/// </param>
/// 
/// <returns>
/// bool type,
/// indicates whether the two cities must be connected or not
/// </returns>
bool IsMustConnect(int node1, int node2, const std::vector<New_Node>& nodes) {
	if (nodes[node1].sign == HeadOfThreeNodes && node2 == node1 + 1)	return true;
	if (nodes[node1].sign == TailOfThreeNodes && node2 == node1 - 1)	return true;

	return false;
}

