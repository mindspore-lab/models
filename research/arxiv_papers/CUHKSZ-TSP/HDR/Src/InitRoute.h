/*
* Name        : InitRoute.h
* Author      : Sipeng, Sun
* Description : This file contails some function about initialing the whole tour
*/


#ifndef _InitRoute_h
#define _InitRoute_h

#include <vector>

#include "TwoOpt.h"
using namespace std;

/// <summary>
/// according to center cities, classify the rest cities
/// </summary>
/// 
/// <param name="in_centers_nodes">
/// vector<vector<int>> type,
/// which stores the classified results
/// </param>
/// 
/// <param name="points">
/// Point array,
/// which stores the information about cities
/// </param>
/// 
/// <param name="n_point">
/// int type,
/// the size of points
/// </param>
/// 
/// <param name="centers">
/// int array,
/// which stores the center cities
/// </param>
/// 
/// <param name="n_center">
/// int type,
/// the size of center cities
/// </param>
/// 
/// <param name="nodes">
/// vector of New_Node,
/// which stores the information of cities after compression
/// </param>
void Classify(vector<vector<int>>& in_centers_nodes, Point* points, int n_point,
	int* centers, int n_center,
	const vector<New_Node>& nodes);
/// <summary>
/// select center cities
/// </summary>
/// 
/// <param name="points">
/// Point array,
/// which stores the information about cities
/// </param>
/// 
/// <param name="n_point">
/// int type,
/// the size of points
/// </param>
/// 
/// <param name="centers">
/// int array,
/// which stores the center cities
/// </param>
/// 
/// <param name="in_centers_nodes">
/// vector<vector<int>> type,
/// which stores the cities in each center
/// </param>
/// 
/// <param name="n_center">
/// int type,
/// the number of center cities
/// </param>
/// 
/// <param name="nodes">
/// vector of New_Node,
/// which stores the information of cities after compression
/// </param>
void SelectCenter(Point* points, int n_point,
	int* centers, vector<vector<int>>& in_centers_nodes, int n_center,
	const vector<New_Node>& nodes);
/// <summary>
/// initial a whole tour
/// </summary>
/// 
/// <param name="points">
/// Point array,
/// which stores the information about cities
/// </param>
/// 
/// <param name="n_point">
/// int type,
/// the size of points
/// </param>
/// 
/// <param name="n_center">
/// int type,
/// the number of center cities
/// </param>
/// 
/// <param name="nodes">
/// vector of New_Node,
/// which stores the information of cities after compression
/// </param>
/// 
/// <returns>
/// int array,
/// the initialed tour
/// </returns>
int* InitPath(Point* points, int n_point, int n_center, const vector<New_Node>& nodes);


#endif // !_InitRoute_h
