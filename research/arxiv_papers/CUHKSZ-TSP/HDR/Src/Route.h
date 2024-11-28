/*
* Name        : Route.h
* Author      : Sipeng, Sun
* Description : This file contails the main structures and some functions about the route.
*/


#ifndef _Route_h
#define _Route_h

#include <string>
#include "Parameter.h"


// the type of nodes after compression
typedef enum {
	Normal, HeadOfThreeNodes, TailOfThreeNodes
}NewNodeKind;

struct Point{
	double x;
	double y;
};	// the struct to stores the coordinates of cities
struct New_Node {
	NewNodeKind sign;	// the node type
	int realID;			// the real city it maps
	std::vector<int> inpath;	// the temporary disappeared cities
};	// the struct to stores the information about cities after compression
struct Edge {
	int first;
	int second;

	bool operator<(const Edge& t) const {
		if (first < t.first)	return true;
		else if (first == t.first && second < t.second) return true;

		return false;
	}
};	// the struct to represent edges
struct DInt {
	int key;
	int value;

	bool operator<(const DInt& t) const {
		return value < t.value;
	}
};	// the struct is mainly used to help sort


typedef long Type_Route_Length;		// the type of route length
const double PI = 3.14159265358979323846264;	// pi value that will be used  to compute distance if problem is world tsp

extern int TSP_SIZE;		// the size of the original instance 
extern std::string INSTANCE;	// the name of the instance
extern std::string INSTANCE_TYPE;	// the type of the instance


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
int PRE_POS(int now_pos, int size, int move_step = 1);
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
int NEXT_POS(int now_pos, int size, int move_step = 1);

/// <summary>
/// read information about the problem from the input file
/// </summary>
/// 
/// <param name="points">
/// Point array, 
/// stores the information about the cordinates of cities 
/// </param>
void ReadProblem(Point*& points);
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
void WriteTour(int* route, Type_Route_Length& Opt);


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
int EUC_2D(Point point1, Point point2);
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
int MAN_2D(Point point1, Point point2);
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
int MAX_2D(Point point1, Point point2);
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
int GEO(Point point1, Point point2);
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
int ATT(Point point1, Point point2);
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
int CEIL_2D(Point point1, Point point2);
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
int WORLD(Point point1, Point point2);
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
int DIS(Point point1, Point point2);

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
Type_Route_Length LENGTH(int* route, Point* points, int n_element);
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
Type_Route_Length LENGTH(int* route, int** DisMatrix, int n_element);

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
bool IsMustConnect(int node1, int node2, const std::vector<New_Node>& nodes);


#endif // !_Route_h