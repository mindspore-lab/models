/*
* Name        : SmallerTSP.cpp
* Author      : Sipeng, Sun
* Description : This file implements the functions in SmallerTSP.h
*/


#include <iostream>
#include <fstream>
#include <climits>
#include <vector>
#include <map>

#include "SmallerTSP.h"

/// <summary>
/// let the appear time of the edge add 1
/// </summary>
/// 
/// <param name="edge_inf">
/// map<Edge, int> type,
/// which stores the appear times of edges
/// </param>
/// 
/// <param name="node1">
/// int type,
/// one endpoint of edge
/// </param>
/// 
/// <param name="node2">
/// int type,
/// one endpoint of edges
/// </param>
void SetEdgeInf(std::map<Edge, int>& edge_inf, int node1, int node2) {
	if (node1 > node2) return SetEdgeInf(edge_inf, node2, node1);

	Edge edge{ node1, node2 };
	edge_inf[edge]++;
}
/// <summary>
/// obtain the appear time of the edge
/// </summary>
/// 
/// <param name="edge_inf">
/// map<Edge, int> type,
/// which stores the appear times of edges
/// </param>
/// 
/// <param name="node1">
/// int type,
/// one endpoint of edge
/// </param>
/// 
/// <param name="node2">
/// int type,
/// one endpoint of edges
/// </param>
/// 
/// <returns>
/// int type,
/// the appear time of the edge
/// </returns>
int GetEdgeInf(std::map<Edge, int>& edge_inf, int node1, int node2) {
	if (node1 > node2) return GetEdgeInf(edge_inf, node2, node1);

	Edge edge{ node1, node2 };
	std::map<Edge, int>::iterator iter = edge_inf.find(edge);
	if (iter != edge_inf.end())	return (*iter).second;

	return 0;
}

/// <summary>
/// obtain the first edge whose appear time is less than one value
/// </summary>
/// 
/// <param name="route">
/// int array,
/// which stores the tour
/// </param>
/// 
/// <param name="s_route">
/// int type,
/// the size of the tour
/// </param>
/// 
/// <param name="edge_inf">
/// map<Edge, int> type,
/// which stores the appear times of edges
/// </param>
/// 
/// <param name="prab">
/// int type,
/// indicates the value
/// </param>
/// 
/// <returns>
/// int type,
/// the second endpoint of the edge
/// </returns>
int GetFirstFreeEdge(int* route, int s_route, std::map<Edge, int>& edge_inf, int prab) {
	for (int i = 0; i < s_route; i++) {
		int start_node = route[i];
		int end_node = route[NEXT_POS(i, s_route)];

		if (GetEdgeInf(edge_inf, start_node, end_node) < prab) return NEXT_POS(i, s_route);
	}
	return -1;
}
/// <summary>
/// compress the tour into a smaller tsp according to edge information
/// </summary>
/// 
/// <param name="route">
/// int array,
/// the target tour
/// </param>
/// 
/// <param name="s_route">
/// int type,
/// the size of the tour
/// </param>
/// 
/// <param name="edge_inf">
/// map<Edge, int> type,
/// which stores the appear times of edges
/// </param>
/// 
/// <param name="prab">
/// int type,
/// compress the tour based on the value.
/// </param>
/// 
/// <returns>
/// vector<New_Node>
/// stores the information of new nodes after compression 
/// </returns>
std::vector<New_Node> ZipTSP(int* route, int s_route,
	std::map<Edge, int>& edge_inf, int prab) {
	std::vector<New_Node> nodes;
	nodes.reserve(s_route / 2);

	// obtain the first edge whose appear time is less than prab.
	int start_index = GetFirstFreeEdge(route, s_route, edge_inf, prab);
	if (start_index == -1) {
		std::cout << "Note:" << std::endl
			<< "All edge must be connected now!" << std::endl
			<< "Cannot go ahead!" << std::endl;
		exit(EXIT_SUCCESS);
	}

	int count = 0;
	while (count < s_route) {
		int node = route[start_index];
		int next_node = route[NEXT_POS(start_index, s_route)];

		// if the appear time is larger than or equal to prab value,
		// then begin to compress
		if (GetEdgeInf(edge_inf, node, next_node) >= prab) {
			New_Node head_node;
			head_node.sign = HeadOfThreeNodes;
			head_node.realID = node;
			count++;

			start_index = NEXT_POS(start_index, s_route);
			node = next_node;
			next_node = route[NEXT_POS(start_index, s_route)];

			New_Node tail_node;
			tail_node.sign = TailOfThreeNodes;
			while (GetEdgeInf(edge_inf, node, next_node) >= prab) {
				tail_node.inpath.push_back(node);
				count++;

				start_index = NEXT_POS(start_index, s_route);
				node = next_node;
				next_node = route[NEXT_POS(start_index, s_route)];
			}
			tail_node.realID = node;
			count++;

			nodes.push_back(head_node);
			nodes.push_back(tail_node);
		}
		else {
			New_Node free_node;
			free_node.sign = Normal;
			free_node.realID = node;
			nodes.push_back(free_node);
			count++;
		}

		start_index = NEXT_POS(start_index, s_route);
	}

	return nodes;
}

/// <summary>
/// obtain the smaller value in a vector
/// </summary>
/// 
/// <param name="L">
/// vector<int>,
/// the target vector
/// </param>
/// 
/// <returns>
/// the index of the minimal value
/// </returns>
int GetMin(const std::vector<int>& L) {
	int sL = L.size();

	int index = rand() % sL;
	int min_value = INT_MAX, min_index = -1;

	for (int i = 0; i < sL; i++) {
		if (L[index] < min_value) {
			min_value = L[index];
			min_index = index;
		}
		index = NEXT_POS(index, sL);
	}

	return min_index;
}
/// <summary>
/// delete some edges
/// </summary>
/// 
/// <param name="n_cut">
/// int type,
/// the maximal number of cut edges
/// </param>
/// 
/// <param name="c_cut">
/// int type,
/// the actual number of cut edges
/// </param>
/// 
/// <param name="route">
/// int array,
/// the target tour
/// </param>
/// 
/// <param name="s_route">
/// int type,
/// the size of route
/// </param>
/// 
/// <param name="candidates">
/// vector<DInt>,
/// where we select edges to cut
/// </param>
/// 
/// <param name="nodes">
/// vector<New_Node>,
/// stores the information of cities after compression
/// </param>
/// 
/// <returns>
/// int array,
/// which stores the information of cut edges
/// </returns>
int* RemoveEdge(const int n_cut, int& c_cut,
	int* route, int s_route,
	std::vector<DInt> candidates, const std::vector<New_Node>& nodes) {

	c_cut = 0;
	int* pos_of_cut_edge = new int[n_cut];
	std::vector<int> node_location(s_route);
	for (int i = 0; i < s_route; i++) node_location[route[i]] = i;

	std::vector<bool> is_visit(s_route, false);
	for (DInt candidate : candidates) {
		if (c_cut >= n_cut) break;
		int node = candidate.key;

		int node_pos = node_location[node];
		int pre_pos = PRE_POS(node_pos, s_route);
		int next_pos = NEXT_POS(node_pos, s_route);

		int pre_node = route[pre_pos];
		int next_node = route[next_pos];

		// if the position is not visited and the edge can be removed, then removed
		if (!is_visit[pre_pos] && !IsMustConnect(pre_node, node, nodes)) {
			pos_of_cut_edge[c_cut++] = pre_pos;
			is_visit[pre_pos] = true;
		}

		if (c_cut >= n_cut) break;
		// if the position is not visited and the edge can be removed, then removed
		if (!is_visit[node_pos] && !IsMustConnect(node, next_node, nodes)) {
			pos_of_cut_edge[c_cut++] = node_pos;
			is_visit[node_pos] = true;
		}
	}

	return pos_of_cut_edge;
}
/// <summary>
/// compress the tour into a smaller tsp according to cut edges
/// </summary>
/// 
/// <param name="n_node">
/// int type,
/// the size of new tsp 
/// </param>
/// 
/// <param name="route">
/// int array,
/// the target tour
/// </param>
/// 
/// <param name="s_route">
/// int type,
/// the size of tour
/// </param>
/// 
/// <param name="pos_of_cut_edge">
/// int array,
/// which store the information of cut edges
/// </param>
/// 
/// <param name="n_cut">
/// int type,
/// the number of cut edges
/// </param>
/// 
/// <returns>
/// vector<New_Node>,
/// stores the information of cities after compression
/// </returns>
std::vector<New_Node> ZipTSP(int& n_node,
	int* route, int s_route,
	int* pos_of_cut_edge, int n_cut) {
	std::vector<New_Node> nodes(n_cut * 2);
	n_node = 0;

	for (int i = 0; i < n_cut; i++) {
		// the start position of the segment
		int start_pos = NEXT_POS(pos_of_cut_edge[i], s_route);
		// the size of the segment
		int s_seg = (pos_of_cut_edge[NEXT_POS(i, n_cut)] - pos_of_cut_edge[i] + s_route) % s_route;

		if (s_seg <= 2) {
			// if the size of segment is less than 3, no change happen
			for (int j = 0; j < s_seg; j++) {
				nodes[n_node].sign = Normal;
				nodes[n_node++].realID = route[NEXT_POS(start_pos, s_route, j)];
			}
		}
		else {
			// if the size is larger than 2, we need to compress the segment into two nodes,
			// one type is HeadOfThreeNodes, and the others is TailOfThreeNodes
			// the cities between the two endpoints will be store in the struct of the city whose type is TailOfThreeNode
			nodes[n_node].sign = HeadOfThreeNodes;
			nodes[n_node++].realID = route[start_pos];

			nodes[n_node].sign = TailOfThreeNodes;
			nodes[n_node].inpath.resize(s_seg - 2);
			for (int j = 0; j < s_seg - 2; j++)
				nodes[n_node].inpath[j] = route[NEXT_POS(start_pos, s_route, j + 1)];
			nodes[n_node++].realID = route[NEXT_POS(start_pos, s_route, s_seg - 1)];
		}
	}

	return nodes;
}
/// <summary>
/// obtain the distance matrix
/// </summary>
/// 
/// <param name="new_nodes">
/// vector<New_Node>,
/// stores the information of cities after two compression
/// </param>
/// 
/// <param name="n_new_nodes">
/// int type,
/// the size of new_nodes
/// </param>
/// 
/// <param name="points">
/// Point type,
/// stores the coordinates of cities
/// </param>
/// 
/// <param name="ori_nodes">
/// vector<New_Node>,
/// stores the information of cities after one compression
/// </param>
/// 
/// <returns>
/// two dimensional int array,
/// the distance matrix
/// </returns>
int** GetDisMatrix(const std::vector<New_Node>& new_nodes, int n_new_nodes,
	Point* points, const std::vector<New_Node>& ori_nodes) {

	int** dis_matrix = new int* [n_new_nodes];
	for (int i = 0; i < n_new_nodes; i++) dis_matrix[i] = new int[n_new_nodes];

	for (int i = 0; i < n_new_nodes; i++) {
		for (int j = i; j < n_new_nodes; j++) {
			int ori_node1 = new_nodes[i].realID;
			int ori_node2 = new_nodes[j].realID;

			// judge whether must connects, if so, distance will be -2, otherwise it is normal
			if (IsMustConnect(i, j, new_nodes) || IsMustConnect(ori_node1, ori_node2, ori_nodes))
				dis_matrix[i][j] = dis_matrix[j][i] = -2;
			else
				dis_matrix[i][j] = dis_matrix[j][i] = DIS(points[ori_node1], points[ori_node2]);
		}
	}

	return dis_matrix;
}
/// <summary>
/// uncompress the tour
/// </summary>
/// 
/// <param name="final_route">
/// int array,
/// the uncompressed tour
/// </param>
/// 
/// <param name="route">
/// int array,
/// the target tour
/// </param>
/// 
/// <param name="s_route">
/// int type,
/// the size of route
/// </param>
/// 
/// <param name="nodes">
/// vector<New_Node>,
/// stores the information of cities after compression
/// </param>
void UnzipTSP(int* final_route, int* route, int s_route,
	const std::vector<New_Node>& nodes) {
	int count = 0;

	for (int i = 0; i < s_route; i++) {
		int node = route[i];

		if (nodes[node].sign != TailOfThreeNodes) 	final_route[count++] = nodes[node].realID;
		else {
			int pre_node = route[PRE_POS(i, s_route)];
			int next_node = route[NEXT_POS(i, s_route)];

			// judge whether the pre city or the next city is the city that it must connects.
			// if neither not, it means some wrong happen 
			if (pre_node == node - 1) {
				for (int n : nodes[node].inpath)	final_route[count++] = n;
				final_route[count++] = nodes[node].realID;
			}
			else if (next_node == node - 1) {
				final_route[count++] = nodes[node].realID;
				for (int j = nodes[node].inpath.size() - 1; j >= 0; j--)
					final_route[count++] = nodes[node].inpath[j];
			}
			else {
				std::cout << "ERROR" << std::endl
					<< "FAIL TO UNZIP!" << std::endl;
				exit(EXIT_FAILURE);
			}
		}
	}
}