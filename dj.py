def dijkstra(adjMatrix, startNode, endNode):
    n = len(adjMatrix)  # 节点数量
    dist = [float('inf')] * n  # 初始化距离为无穷大
    prev = [-1] * n  # 前驱节点数组（用-1表示无前驱）
    visited = [False] * n  # 访问标记数组
    
    dist[startNode] = 0  # 起点距离设为0
    
    for i in range(n):
        # 找到未访问节点中距离最小的节点
        minDist = float('inf')
        u = -1
        for v in range(n):
            if not visited[v] and dist[v] < minDist:
                minDist = dist[v]
                u = v
        
        # 所有节点已访问或剩余节点不可达
        if u == -1 or u == endNode:
            break
        
        visited[u] = True  # 标记为已访问
        
        # 更新邻居节点的距离
        for v in range(n):
            edge_weight = adjMatrix[u][v]
            if edge_weight > 0:  # 存在有效边
                alt = dist[u] + edge_weight
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
    
    # 回溯构建路径
    path = []
    if dist[endNode] < float('inf'):
        u = endNode
        while u != -1:
            path.insert(0, u)  # 在头部插入节点
            u = prev[u]
        minCost = dist[endNode]
    else:
        minCost = float('inf')
        path = []
    
    return minCost, path
