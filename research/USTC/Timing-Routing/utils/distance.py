def get_distance(graph: 'list[list[tuple]]', root: tuple):
    '''
    输入：邻接表，根节点（都是坐标）
    输出：每个点到根节点的距离 字典
    '''

    ans = {root: 0}
    parent = {root: root}
    graph_new = {}
    for edge in graph:
        if edge[0] not in graph_new:
            graph_new[edge[0]] = []
        if edge[1] not in graph_new:
            graph_new[edge[1]] = []
        graph_new[edge[0]].append((edge[1], 0))  # 元组第二位表示边的方向
        graph_new[edge[1]].append((edge[0], 1))  # 元组第二位表示边的方向

    def process_parent(point):
        '''
        处理路径上的重叠，返回重叠长度
        '''
        process_ans = 0
        if parent[point] not in ans:
            process_ans += process_parent(parent[point])
        p = parent[point]
        pp = parent[p]
        if point[0] == p[0] and p[0] == pp[0]:
            process_ans += abs(point[1] - p[1]) + abs(p[1] - pp[1]) - abs(point[1] - pp[1])
            parent[point] = pp
        elif point[1] == p[1] and p[1] == pp[1]:
            process_ans += abs(point[0] - p[0]) + abs(p[0] - pp[0]) - abs(point[0] - pp[0])
            parent[point] = pp
        if len(p) == 3:
            if parent[point] != pp:
                parent[point] = p[:2]
                parent[p[:2]] = pp
            del parent[p]
        if pp == point[:2]:
            parent[point] = parent[pp]
        return process_ans

    max_length = 0

    def dfs(root):
        for point, direction in graph_new[root]:
            if point in ans or point == root:
                continue  # 处理过
            tmp_point = (root[0], point[1], 1) if direction == 0 else (point[0], root[1], 1)
            if tmp_point == root or tmp_point == point:
                parent[point] = root
            else:
                parent[point] = tmp_point
                parent[tmp_point] = root
            ans[point] = ans[root] + abs(root[0] - point[0]) + abs(root[1] - point[1]) - process_parent(point)
            nonlocal max_length
            max_length = max(max_length, ans[point])
            dfs(point)
        return
    dfs(root)
    # print(parent)
    return ans, max_length


if __name__ == '__main__':
    # a = [(0, 2), (0, 6), (1, 2), (5, 6), (5, 3), (4, 3)]  # edge
    # b = [(3, 4), (4, 1), (5, 5), (0, 1), (1, 0), (2, 7), (6, 6)]  # points
    a = [(1, 0), (2, 0)]
    b = [(1, 2), (2, 1), (3, 3)]
    print(get_distance([[b[i], b[j]] for i, j in a], b[0]))
