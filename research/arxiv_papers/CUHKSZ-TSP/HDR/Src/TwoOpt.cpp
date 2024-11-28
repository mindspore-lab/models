/*
* Name        : TwoOpt.cpp
* Author      : Sipeng, Sun
* Description : This file implements the functions in TwoOpt.h
*/


#include <vector>
#include <queue>

#include "TwoOpt.h"
using namespace std;

/// <summary>
/// initial and mainly allocating spaces
/// </summary>
/// 
/// <param name="n_city">
/// int type,
/// the size of cities
/// </param>
TwoOpt::TwoOpt(int n_city) {
	this->n_city = n_city;


	dis_matrix = new int* [n_city];
	candidates = new int* [n_city];
	for (int i = 0; i < n_city; i++) {
		dis_matrix[i] = new int[n_city];
		candidates[i] = new int[CANDIDATE_NUM];
	}

	candidate_size = new int[n_city];
}
/// <summary>
/// release all the allocated memory
/// </summary>
TwoOpt::~TwoOpt() {
	for (int i = 0; i < n_city; i++) {
		delete[] dis_matrix[i];
		delete[] candidates[i];
	}
	delete[] dis_matrix;
	delete[] candidates;

	delete[] candidate_size;
}

/// <summary>
/// compute the distance matrix
/// </summary>
/// 
/// <param name="points">
/// Point type,
/// which stores the information about cities
/// </param>
/// 
/// <param name="real_id">
/// int array,
/// which indicates what city each city maps to
/// </param>
/// 
/// <param name="nodes">
/// vector of New_Node,
/// which stores the information of cities after compression
/// </param>
void TwoOpt::GetDisMatrix(Point* points, int* real_id, const std::vector<New_Node>& nodes) {
	for (int i = 0; i < n_city; i++) {
		int real_node1 = real_id[i];
		for (int j = i; j < n_city; j++) {
			int real_node2 = real_id[j];

			// if two cities must be connected, the distance will be -2; Otherwise, the distance is computed by DIS function.
			if (IsMustConnect(real_node1, real_node2, nodes)) dis_matrix[i][j] = dis_matrix[j][i] = -2;
			else dis_matrix[i][j] = dis_matrix[j][i] = DIS(points[real_node1], points[real_node2]);
		}
	}
}
/// <summary>
/// obtain the nearest cities from the target city.
/// </summary>
/// 
/// <param name="node">
/// int type,
/// which is the target city.
/// </param>
void TwoOpt::GetCandidate(int node) {
	// use priority queue to obtain the nearest cities
	std::priority_queue<DInt> heap;
	for (int i = 0; i < n_city; i++) {
		if (i == node)	continue;

		int dis = dis_matrix[node][i];
		DInt element{ i, dis };
		if (heap.size() < CANDIDATE_NUM || element < heap.top()) heap.push(element);
		if (heap.size() > CANDIDATE_NUM) heap.pop();
	}

	// use our varables to store the information
	candidate_size[node] = heap.size();
	for (int i = heap.size() - 1; i >= 0; i--) {
		candidates[node][i] = heap.top().key;
		heap.pop();
	}
}

/// <summary>
/// initial a tour by using variant greedy algorithm 
/// </summary>
/// 
/// <param name="route">
/// int array,
/// which stores the tour
/// </param>
/// 
/// <param name="all_nodes_pos">
/// int array,
/// which stores the position of each city
/// </param>
/// 
/// <returns>
/// Type_Route_Length type,
/// the length of tour after initial
/// </returns>
Type_Route_Length TwoOpt::GreedyInitRoute(int* route, int* all_nodes_pos) {
	Type_Route_Length length = 0;

	std::vector<bool> is_visit(n_city, false);
	route[0] = 0;
	all_nodes_pos[0] = 0;
	is_visit[0] = true;

	for (int pos = 1; pos < n_city; pos++) {
		int pre_node = route[pos - 1];

		// obtain the nearest 5 unvisited cities 
		std::priority_queue<DInt> heap;
		for (int i = 0; i < n_city; i++) {
			if (is_visit[i])	continue;

			int dis = dis_matrix[pre_node][i];
			DInt element{ i, dis };
			if (heap.size() < 5 || element < heap.top()) heap.push(element);
			if (heap.size() > 5) heap.pop();
		}

		// random select one city with different probilities 
		int random_select;
		if (heap.size() == 5) {
			random_select = rand() % 16;

			if (random_select <= 7) random_select = 4;
			else if (random_select <= 11) random_select = 3;
			else if (random_select <= 13) random_select = 2;
			else if (random_select <= 14) random_select = 1;
			else random_select = 0;
		}
		else random_select = rand() % heap.size();

		for (int i = 0; i < random_select; i++) heap.pop();
		random_select = heap.top().key;

		route[pos] = random_select;
		all_nodes_pos[random_select] = pos;
		is_visit[random_select] = true;
		length += heap.top().value;
	}

	length += dis_matrix[route[n_city - 1]][route[0]];
	return length;
}

/// <summary>
/// partly reverse a tour
/// </summary>
/// 
/// <param name="route">
/// int array,
/// which is the target tour
/// </param>
/// 
/// <param name="start_pos">
/// int type,
/// indicates the start position to reverse
/// </param>
/// 
/// <param name="end_pos">
/// int type,
/// indicates the end position to reverse
/// </param>
/// 
/// <param name="all_nodes_pos">
/// int array,
/// stores the position of each city
/// </param>
void TwoOpt::ReversePartly(int* route, int start_pos, int end_pos, int* all_nodes_pos) {
	int k = (end_pos - start_pos + 1) / 2;
	for (int i = 0; i < k; i++) {
		int pos1 = start_pos + i, pos2 = end_pos - i;
		int node1 = route[pos1], node2 = route[pos2];

		all_nodes_pos[node1] = pos2;
		all_nodes_pos[node2] = pos1;

		route[pos1] = node2;
		route[pos2] = node1;
	}
}
/// <summary>
/// two opt operation
/// </summary>
/// 
/// <param name="route">
/// int array,
/// which is the target tour
/// </param>
/// 
/// <param name="sum">
/// Type_Route_Lengh type,
/// which is the length of previous tour
/// </param>
/// 
/// <param name="all_nodes_pos">
/// int array,
/// stores the position of each city
/// </param>
/// 
/// <returns>
/// Type_Route_Lengh type,
/// the length of optimised tour
/// </returns>
Type_Route_Length TwoOpt::twoOpt(int* route, Type_Route_Length sum, int* all_nodes_pos) {
	int x1, x2, y1, y2, d;
	for (int i = 0; i < n_city - 2; i++) {
		int node = route[PRE_POS(i, n_city)];

		// select city from the candidate
		for (int j = 0; j < candidate_size[node]; j++) {
			int pos = all_nodes_pos[candidates[node][j]];

			if (pos > i && pos < n_city - 1) {
				x1 = node;
				x2 = route[i];
				y1 = route[pos];
				y2 = route[pos + 1];

				// if the distance is -2, which means the two cities must be connnected, 
				// then they cannot manipulate two-opt operation 
				if (dis_matrix[x1][x2] == -2 || dis_matrix[y1][y2] == -2) continue;

				d = dis_matrix[x1][x2] + dis_matrix[y1][y2]
					- dis_matrix[x1][y1] - dis_matrix[x2][y2];

				// if the tour can be improved after two-opt, it manipulates two-opt operation 
				if (d > 0) {
					ReversePartly(route, i, pos, all_nodes_pos);
					sum -= d;
				}
			}
		}
	}
	return sum;
}

/// <summary>
/// random disturb the tour, ignoring the information that some couple cities must be connected
/// </summary>
/// 
/// <param name="route">
/// int array,
/// which is the target tour
/// </param>
/// 
/// <param name="num">
/// int type,
/// the number of disturbance
/// </param>
void TwoOpt::Disturb(int* route, int num) {
	for (int i = 0; i < num; i++) {
		int pos1 = rand() % n_city;
		int pos2 = rand() % n_city;
		if (pos1 == pos2) { i--; continue; }

		int temp = route[pos1];
		route[pos1] = route[pos2];
		route[pos2] = temp;

		return;
	}
}
/// <summary>
/// since there are several couple cities that must be connected,
/// the program needs rearrange the tour to make these couple cities connected
/// </summary>
/// 
/// <param name="route">
/// int array,
/// which is the tour before rearrangement
/// </param>
/// 
/// <param name="new_route">
/// int array,
/// which is the tour after rearrangement
/// </param>
/// 
/// <param name="new_all_nodes_pos">
/// int array,
/// stores the position of each city in new_route
/// </param>
void TwoOpt::Rearrange(int* route, int* new_route, int* new_all_nodes_pos) {
	int count = 0;
	std::vector<bool> is_visit(n_city, false);

	for (int i = 0; i < n_city; i++) {
		int node = route[i];
		if (is_visit[node])	continue;

		new_route[count] = node;
		new_all_nodes_pos[node] = count++;
		is_visit[node] = true;

		// first judge whether it has the city that must be connected.
		// if so, put the city on the next position
		if (dis_matrix[node][candidates[node][0]] == -2) {
			node = candidates[node][0];

			new_route[count] = node;
			new_all_nodes_pos[node] = count++;
			is_visit[node] = true;
		}
	}
	return;
}


/// <summary>
/// use two-opt to optimise the tour
/// </summary>
/// 
/// <param name="route">
/// int array,
/// which is the target tour
/// </param>
/// 
/// <param name="s_route">
/// int type,
/// which is the size of tour
/// </param>
/// 
/// <param name="points">
/// Point type,
/// which stores the information about cities
/// </param>
/// 
/// <param name="real_id">
/// int array,
/// which indicates what city each city maps to
/// </param>
/// 
/// <param name="nodes">
/// vector of New_Node,
/// which stores the information of cities after compression
/// </param>
/// 
/// <param name="head_connects_tail">
/// bool type,
/// which controls whether the tour satisfies the condition that head connectes tail
/// (head: the minimal index;  tail: the maximal index)
/// </param>
/// 
/// <param name="init">
/// bool type,
/// which controls whether initial a tour
/// </param>
/// 
/// <param name="n_disturb">
/// int type,
/// the number of disturbance
/// </param>
/// 
/// <returns>
/// Type_Route_Length type,
/// the length of optimised tour
/// </returns>
Type_Route_Length TwoOptOptimise(int*& route, int s_route, Point* points, int* real_id, const std::vector<New_Node>& nodes,
	bool head_connects_tail, bool init, int n_disturb) {
	int* all_nodes_pos = new int[s_route];

	TwoOpt two_opt_solver(s_route);
	two_opt_solver.GetDisMatrix(points, real_id, nodes);
	if (head_connects_tail) two_opt_solver.dis_matrix[0][s_route - 1] =
		two_opt_solver.dis_matrix[s_route - 1][0] = -2;
	for (int i = 0; i < s_route; i++) two_opt_solver.GetCandidate(i);

	Type_Route_Length old_length, length;
	if (init)	length = two_opt_solver.GreedyInitRoute(route, all_nodes_pos);
	else {
		for (int i = 0; i < s_route; i++) all_nodes_pos[i] = route[i] = i;
		length = LENGTH(route, two_opt_solver.dis_matrix, s_route);
	}

	// continue manipulating two-opt operation until no improvement
	do {
		old_length = length;
		length = two_opt_solver.twoOpt(route, length, all_nodes_pos);
	} while (old_length != length);

	// disturb
	for (int i = 0; i < n_disturb; i++) {
		int* old_route = new int[s_route];
		int* new_route = new int[s_route];
		int* new_all_nodes_pos = new int[s_route];
		for (int j = 0; j < s_route; j++) {
			old_route[j] = new_route[j] = route[j];
			new_all_nodes_pos[j] = all_nodes_pos[j];
		}

		two_opt_solver.Disturb(old_route, s_route / 20);
		two_opt_solver.Rearrange(old_route, new_route, new_all_nodes_pos);

		Type_Route_Length new_length = LENGTH(new_route, two_opt_solver.dis_matrix, s_route);
		do {
			old_length = new_length;
			new_length = two_opt_solver.twoOpt(new_route, new_length, new_all_nodes_pos);
		} while (old_length != new_length);

		// if the solution is better, then replace 
		if (new_length < length) {
			length = new_length;

			delete[] old_route;
			delete[] route;
			delete[] all_nodes_pos;

			route = new_route;
			all_nodes_pos = new_all_nodes_pos;
		}
		else {
			delete[] old_route;
			delete[] new_route;
			delete[] new_all_nodes_pos;
		}
	}

	delete[] all_nodes_pos;
	return length;
}