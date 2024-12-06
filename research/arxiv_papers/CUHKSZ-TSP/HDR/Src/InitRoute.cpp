/*
* Name        : InitRoute.cpp
* Author      : Sipeng, Sun
* Description : This file implements the functions in InitRoute.h
*/


#include <iostream>

#include "InitRoute.h"


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
	const vector<New_Node>& nodes) {
	vector<bool> is_visit(n_point, false);

	// delete center cities, and visit the cities that must be connected with those cities
	for (int i = 0; i < n_center; i++) {
		int center = centers[i];
		is_visit[center] = true;

		if (nodes[center].sign == HeadOfThreeNodes) {
			in_centers_nodes[i].push_back(center + 1);
			is_visit[center + 1] = true;
		}
		else if (nodes[center].sign == TailOfThreeNodes) {
			in_centers_nodes[i].push_back(center - 1);
			is_visit[center - 1] = true;
		}
	}

	// judge which center each city belong to according to distance
	for (int i = 0; i < n_point; i++) {
		if (is_visit[i]) continue;

		int belong = 0, dis = DIS(points[i], points[centers[0]]);
		for (int j = 1; j < n_center; j++) {
			int new_dis = DIS(points[i], points[centers[j]]);
			if (new_dis < dis) { belong = j; dis = new_dis; }
		}

		in_centers_nodes[belong].push_back(i);
		is_visit[i] = true;

		// it has must connected city, then add
		if (nodes[i].sign == HeadOfThreeNodes) {
			in_centers_nodes[belong].push_back(i + 1);
			is_visit[i + 1] = true;
		}
	}
}
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
	const vector<New_Node>& nodes) {
	bool is_fail = true;
	while (is_fail) {
		is_fail = false;
		for (int i = 0; i < n_center; i++) { in_centers_nodes[i].clear(); }

		// random select center cities
		for (int i = 0; i < n_center; i++) {
			int r = rand() % n_point;
			centers[i] = r;

			for (int j = 0; j < i; j++) {
				// center cities should be discrete and should not be connected
				if (centers[j] == r || IsMustConnect(centers[j], r, nodes)) {
					i--;
					break;
				}
			}
		}

		// classify all cities
		Classify(in_centers_nodes, points, n_point, centers, n_center, nodes);
		// the size of each center should not be 0
		for (int i = 0; i < n_center; i++) {
			if (!in_centers_nodes[i].size()) { is_fail = true; break; }
		}
	}
}
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
int* InitPath(Point* points, int n_point, int n_center, const vector<New_Node>& nodes) {
	int* route = new int[n_point];

	int* centers = new int[n_center];
	int* centers_route = new int[n_center];

	vector<vector<int>> in_centers_nodes(n_center);
	for (int i = 0; i < n_center; i++) in_centers_nodes[i].reserve(n_point / n_center);

	// random select center cities
	SelectCenter(points, n_point, centers, in_centers_nodes, n_center, nodes);
	// get an optimised tour of center cities
	TwoOptOptimise(centers_route, n_center, points, centers, nodes, false, true, 500);

	// insert the rest cities after each center cities
	int c_route = 0;
	for (int i = 0; i < n_center; i++) {
		int center = centers_route[i];

		route[c_route++] = centers[center];
		for (int node : in_centers_nodes[center]) route[c_route++] = node;
	}


	int count_pos = 0;
	// optimise two adjacent centers by two-opt
	for (int i = 0; i < n_center; i++) {
		int start_center = centers[centers_route[i]];
		int end_center = centers[centers_route[NEXT_POS(i, n_center, 2)]];

		while (route[count_pos] != start_center) count_pos = NEXT_POS(count_pos, n_point);
		if (IsMustConnect(start_center, route[NEXT_POS(count_pos, n_point)], nodes))
			count_pos = NEXT_POS(count_pos, n_point);

		int start_pos = count_pos;
		int s_optimise = 0;
		while (route[count_pos] != end_center) {
			count_pos = NEXT_POS(count_pos, n_point); s_optimise++;
		}
		if (!IsMustConnect(end_center, route[PRE_POS(count_pos, n_point)], nodes))
			s_optimise++;

		int* real_id = new int[s_optimise];
		int* optimise_route = new int[s_optimise];

		for (int j = 0; j < s_optimise; j++) real_id[j] = route[NEXT_POS(start_pos, n_point, j)];
		TwoOptOptimise(optimise_route, s_optimise, points, real_id, nodes, true, false, 100);


		int pos;
		for (pos = 0; pos < s_optimise; pos++) {
			if (!optimise_route[pos]) break;
		}

		if (optimise_route[NEXT_POS(pos, s_optimise)] == (s_optimise - 1)) {
			for (int j = 0; j < s_optimise; j++) {
				route[NEXT_POS(start_pos, n_point, j)] = real_id[optimise_route[PRE_POS(pos, s_optimise, j)]];
			}
		}
		else if (optimise_route[PRE_POS(pos, s_optimise)] == (s_optimise - 1)) {
			for (int j = 0; j < s_optimise; j++) {
				route[NEXT_POS(start_pos, n_point, j)] = real_id[optimise_route[NEXT_POS(pos, s_optimise, j)]];
			}
		}
		else {
			printf("\nERROR\n");
			exit(1);
		}


		delete[] real_id;
		delete[] optimise_route;
	}

	delete[] centers;
	delete[] centers_route;


	return route;
}

