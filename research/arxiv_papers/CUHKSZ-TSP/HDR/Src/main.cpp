#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <climits>

#include "Parameter.h"
#include "Route.h"
#include "TwoOpt.h"
#include "InitRoute.h"
#include "SmallerTSP.h"
#include "Solve_subTSP/Solve_Sub_TSP.h"
using namespace std;



clock_t start_time = 0;	// the time when program starts
Point* ori_points = NULL;
Point* now_points = NULL;
int* solution = NULL;
int** all_solutions = NULL;	
Type_Route_Length best_length = LONG_MAX;

/// <summary>
/// obtain the time difference
/// </summary>
/// 
/// <param name="time1">
/// clock_t type,
/// the start time
/// </param>
/// 
/// <param name="time2">
/// clock_t type,
/// the end time
/// </param>
/// 
/// <returns>
/// double type,
/// the difference of the two time
/// </returns>
double GetTime(clock_t time1, clock_t time2) {
	return (double)(time2 - time1) / CLOCKS_PER_SEC;
}

/// <summary>
/// allocate memory for some global variables
/// </summary>
void Allocate() {
	all_solutions = new int* [SOLUTION_NUM];
	for (int i = 0; i < SOLUTION_NUM; i++) all_solutions[i] = new int[TSP_SIZE];
}
/// <summary>
/// release momory for global variables
/// </summary>
void Destroy() {
	delete[] ori_points;

	for (int i = 0; i < SOLUTION_NUM; i++) delete[] all_solutions[i];
	delete[] all_solutions;

	if (solution) delete[] solution;
	if (now_points) delete[] now_points;
}
/// <summary>
/// terminate the program
/// </summary>
void Terminate() {
	Destroy();
	printf("Best: %ld  Time : %.2fs\n", best_length, GetTime(start_time, clock()));
	exit(EXIT_SUCCESS);
}

/// <summary>
/// collect edge information
/// </summary>
/// 
/// <returns>
/// map<Edge, int>,
/// stores the information of edges
/// </returns>
map<Edge, int> MergeAllRoutes() {
	map<Edge, int> edge_inf;
	for (int i = 0; i < SOLUTION_NUM; i++) {
		int* temp_route = all_solutions[i];

		for (int j = 0; j < TSP_SIZE; j++) {
			int start_node = temp_route[j];
			int end_node = temp_route[NEXT_POS(j, TSP_SIZE)];

			SetEdgeInf(edge_inf, start_node, end_node);
		}
	}

	return edge_inf;
}
/// <summary>
/// update the best solution
/// </summary>
/// 
/// <param name="route">
/// int array,
/// the best solution
/// </param>
/// 
/// <param name="length">
/// Type_Route_Length type,
/// the length of the best solution
/// </param>
void UpdateTour(int* route, Type_Route_Length& length) {
	cout << "Write Tour : " << OUTPUT_FILE_PATH << endl;
	best_length = length;
	WriteTour(route, best_length);
}
/// <summary>
/// directly solve the tsp problem
/// </summary>
/// 
/// <param name="points">
/// Point array,
/// stores the coordinates of cities
/// </param>
/// 
/// <param name="nodes">
/// vector<New_Node>
/// stores the information about compressed cities 
/// </param>
/// 
/// <returns>
/// int array,
/// the optimised tour
/// </returns>
int* DirectSolveByEax(Point* points, const vector<New_Node>& nodes) {
	int n_node = nodes.size();
	int* route = new int[n_node];

	int** dis_matrix = new int* [n_node];
	for (int i = 0; i < n_node; i++)
		dis_matrix[i] = new int[n_node];

	for (int i = 0; i < n_node; i++) {
		for (int j = i; j < n_node; j++) {
			if (IsMustConnect(i, j, nodes)) dis_matrix[i][j] = dis_matrix[j][i] = -2;
			else dis_matrix[i][j] = dis_matrix[j][i] = DIS(points[i], points[j]);
		}
	}
	int length = Solve_Sub_TSP(n_node, dis_matrix, route);

	for (int i = 0; i < n_node; i++)
		delete[] dis_matrix[i];
	delete[] dis_matrix;
	return route;
}



int main(int argc, char* argv[]) {
	start_time = clock();

	// read data
	ReadParameter(argc, argv);
	ReadProblem(ori_points);
	
	// set seed and allocate memory
	srand(SEED);
	Allocate();
	OutputParameter();

	// At the begining, the compressed cities is the original problem
	map<Edge, int> edge_inf;
	int new_tsp_size = TSP_SIZE;
	vector<New_Node> first_nodes(TSP_SIZE);
	for (int i = 0; i < TSP_SIZE; i++) {
		first_nodes[i].sign = Normal;
		first_nodes[i].realID = i;
	}


	for (int c_iteration = 0; c_iteration < OPERATION_NUM; c_iteration++) {
		// obtain the coordinates of compressed cities
		now_points = new Point[new_tsp_size];
		for (int i = 0; i < new_tsp_size; i++) 
			now_points[i] = ori_points[first_nodes[i].realID];

		// get the number of destory & repair operations
		int n_opeation = new_tsp_size / OPERATION_NUM;
		cout << "new tsp size: " << new_tsp_size << endl;

		if (new_tsp_size <= 4) 	Terminate();
		if (new_tsp_size < 500) {
			// the size is smaller, eax can directly get the optimal.
			solution = DirectSolveByEax(now_points, first_nodes);
			UnzipTSP(all_solutions[0], solution, new_tsp_size, first_nodes);
			Type_Route_Length final_length = LENGTH(all_solutions[0], ori_points, TSP_SIZE);
			printf("RUN (%d) Directly use Eax\nCost : %ld  Time : %.2fs\n\n", c_iteration,
				final_length, GetTime(start_time, clock()));

			if (final_length < best_length) UpdateTour(all_solutions[0], final_length);
			Terminate();
		}

		for (int c_solution = 0; c_solution < SOLUTION_NUM; c_solution++) {
			// initial a feasible solution
			solution = InitPath(now_points, new_tsp_size, sqrt(new_tsp_size), first_nodes);
			UnzipTSP(all_solutions[c_solution], solution, new_tsp_size, first_nodes);	
			Type_Route_Length final_length = LENGTH(all_solutions[c_solution], ori_points, TSP_SIZE);			
			printf("* (%d,%d) Init Cost: %ld  Time : %.2fs\n", c_iteration, c_solution, 
				final_length, GetTime(start_time, clock()));

			vector<int> visit_time(new_tsp_size, 0);
			for (int c_operation = 0; c_operation < n_opeation; c_operation++) {
				// select one city as the center of cut edges
				int target = GetMin(visit_time);
				visit_time[target]++;

				// obtain the candidatea of cut edges (sort cities according to distance)
				vector<DInt> candidate_nodes(new_tsp_size);
				for (int i = 0; i < new_tsp_size; i++) {
					int dis = DIS(now_points[i], now_points[target]);

					candidate_nodes[i] = DInt{ i, dis };
				}
				sort(candidate_nodes.begin(), candidate_nodes.end());

				// remove edges
				int n_cut = 0;
				int* pos_of_cut_edge = RemoveEdge(REMOVE_EDGE_NUM, n_cut, solution, new_tsp_size, candidate_nodes, first_nodes);
				sort(pos_of_cut_edge, pos_of_cut_edge + n_cut);
				for (int i = 0; i < n_cut; i++) {
					int node = solution[pos_of_cut_edge[i]];
					visit_time[node]++;
				}

				// compress the tour into smaller tsp instance
				int n_nodes = 0;
				vector<New_Node> second_nodes = ZipTSP(n_nodes, solution, new_tsp_size, pos_of_cut_edge, n_cut);

				// obtain the distance matrix
				int** dis_matrix = GetDisMatrix(second_nodes, n_nodes, now_points, first_nodes);

				// solve the new tsp instance
				int pre_length = 0;
				int* sub_solution = new int[n_nodes];
				for (int i = 0; i < n_nodes; i++)
					pre_length += abs(dis_matrix[i][NEXT_POS(i, n_nodes)]);
				int now_length = Solve_Sub_TSP(n_nodes, dis_matrix, sub_solution);

				// if the result is better, update the solution
				if (now_length < pre_length) {
					UnzipTSP(solution, sub_solution, n_nodes, second_nodes);
					final_length -= pre_length - now_length;
					printf("* (%d,%d,%d)  Cost : %ld  Time : %.2fs\n", c_iteration, c_solution, c_operation,
						final_length, GetTime(start_time, clock()));
				}

				delete[] pos_of_cut_edge;
				for (int i = 0; i < n_nodes; i++) 
					delete[] dis_matrix[i];
				delete[] dis_matrix;
				delete[] sub_solution;

				// if the running time reaches the limited value, then program terminates
				if (TIME > 0 && GetTime(start_time, clock()) >= TIME - 1) {
					cout << "Time terminate!" << endl;
					UnzipTSP(all_solutions[c_solution], solution, new_tsp_size, first_nodes);
					printf("RUN (%d,%d)  Cost : %ld  Time : %.2fs\n\n", c_iteration, c_solution,
						final_length, GetTime(start_time, clock()));

					if (final_length < best_length) UpdateTour(all_solutions[c_solution], final_length);
					Terminate();
				}
			} //c_operation

			UnzipTSP(all_solutions[c_solution], solution, new_tsp_size, first_nodes);
			printf("RUN (%d,%d)  Cost : %ld  Time : %.2fs\n\n", c_iteration, c_solution,
				final_length, GetTime(start_time, clock()));

			if (final_length < best_length) UpdateTour(all_solutions[c_solution], final_length);
			delete[] solution;
			solution = NULL;
		} // c_solution

		// collect edge information
		edge_inf = MergeAllRoutes();
		// if all solutions have the same path, the program terminates
		if (edge_inf.size() == TSP_SIZE) {
			cout << "Solution termiante!" << endl;			
			Terminate();
		}
		for(int i = 0; i < SOLUTION_NUM; i++){
			if(LENGTH(all_solutions[i], ori_points, TSP_SIZE) != best_length) break;
			else if(i == SOLUTION_NUM - 1) Terminate();
		}

		// compress original problem into smaller tsp instance according to new edge information
		first_nodes = ZipTSP(all_solutions[0], TSP_SIZE, edge_inf, SOLUTION_NUM);
		new_tsp_size = first_nodes.size();

		delete[] now_points;
		now_points = NULL;
	} // c_iteration

	Terminate();
	return 1;
}