/*
* Name        : TwoOpt.h
* Author      : Sipeng, Sun
* Description : This file contails some function about two opt.
*/


#ifndef _TwoOpt_h
#define _TwoOpt_h

#include "Route.h"

class TwoOpt {
private:
	const int CANDIDATE_NUM = 50;	// the maximal size of candidate

	int n_city;		// the numbe of cities
	int** candidates;	
	int* candidate_size;

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
	void ReversePartly(int* route, int start_pos, int end_pos, int* all_nodes_pos);
public:
	int** dis_matrix;

	/// <summary>
	/// initial and mainly allocating spaces
	/// </summary>
	/// 
	/// <param name="n_city">
	/// int type,
	/// the size of cities
	/// </param>
	TwoOpt(int n_city);
	/// <summary>
	/// release all the allocated memory
	/// </summary>
	~TwoOpt();


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
	void GetDisMatrix(Point* points, int* real_id, const std::vector<New_Node>& nodes);
	/// <summary>
	/// obtain the nearest cities from the target city.
	/// </summary>
	/// 
	/// <param name="node">
	/// int type,
	/// which is the target city.
	/// </param>
	void GetCandidate(int node);

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
	Type_Route_Length GreedyInitRoute(int* route, int* all_nodes_pos);
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
	Type_Route_Length twoOpt(int* route, Type_Route_Length now_sum, int* all_nodes_pos);
	
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
	void Disturb(int* route, int num);
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
	void Rearrange(int* route, int* new_route, int* new_all_nodes_pos);
};


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
	bool head_connects_tail, bool init, int n_disturb);

#endif // !_TwoOpt_h
