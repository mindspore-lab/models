/*
* Name        : SmallerTSP.h
* Author      : Sipeng, Sun
* Description : This file contails some functions about the main process.
*/

#ifndef _SmallerTSP_h
#define _SmallerTSP_h

#include "Route.h"


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
void SetEdgeInf(std::map<Edge, int>& edge_inf, int node1, int node2);
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
int GetEdgeInf(std::map<Edge, int>& edge_inf, int node1, int node2);

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
int GetFirstFreeEdge(int* route, int s_route, std::map<Edge, int>& edge_inf, int prab);
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
	std::map<Edge, int>& edge_inf, int prab);

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
int GetMin(const std::vector<int>& L);
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
	std::vector<DInt> candidates, const std::vector<New_Node>& nodes);
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
	int* pos_of_cut_edge, int n_cut);
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
	Point* points, const std::vector<New_Node>& ori_nodes);
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
	const std::vector<New_Node>& nodes);

#endif	//_SmallerTSP_h