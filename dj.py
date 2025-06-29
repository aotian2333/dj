#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import heapq
import numpy as np

class DijkstraPlanner:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('dijkstra_planner', anonymous=True)
        
        # 订阅地图信息和目标位置
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_callback)
        self.current_pose_sub = rospy.Subscriber("/base_pose_ground_truth", PoseStamped, self.pose_callback)
        
        # 发布路径规划结果
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=1)
        
        # 初始化变量
        self.map = None
        self.resolution = None
        self.origin = None
        self.goal = None
        self.current_pose = None
        
    def map_callback(self, data):
        # 获取地图的分辨率和原点，将地图数据转换为numpy数组
        self.resolution = data.info.resolution
        self.origin = (data.info.origin.position.x, data.info.origin.position.y)
        self.map = np.array(data.data).reshape(data.info.height, data.info.width)
        
    def goal_callback(self, data):
        # 获取目标位置，将其转换为地图中的坐标
        self.goal = self.world_to_map((data.pose.position.x, data.pose.position.y))
        
    def pose_callback(self, data):
        # 获取当前位置，将其转换为地图中的坐标
        self.current_pose = self.world_to_map((data.pose.position.x, data.pose.position.y))
        
        # 如果地图、当前位置和目标位置都有了，则开始规划路径
        if self.map is not None and self.current_pose is not None and self.goal is not None:
            path = self.dijkstra(self.current_pose, self.goal)
            self.publish_path(path)
            
    def world_to_map(self, world_point):
        # 将世界坐标转换为地图坐标
        mx = int((world_point[0] - self.origin[0]) / self.resolution)
        my = int((world_point[1] - self.origin[1]) / self.resolution)
        return (mx, my)
    
    def map_to_world(self, map_point):
        # 将地图坐标转换为世界坐标
        wx = map_point[0] * self.resolution + self.origin[0]
        wy = map_point[1] * self.resolution + self.origin[1]
        return (wx, wy)
    
    def dijkstra(self, startNode, endNode):
        n = self.map.shape[0]  # 节点数量（假设地图是一个二维网格）
        dist = [float('inf')] * (n * n)  # 初始化距离为无穷大
        prev = [-1] * (n * n)  # 前驱节点数组（用-1表示无前驱）
        visited = [False] * (n * n)  # 访问标记数组
        
        # 起点距离设为0
        start_index = self.map_to_index(startNode)
        dist[start_index] = 0
        
        # 使用优先队列
        priority_queue = []
        heapq.heappush(priority_queue, (0, startNode))
        
        while priority_queue:
            current_cost, current_node = heapq.heappop(priority_queue)
            
            if current_node == endNode:
                break
            
            current_index = self.map_to_index(current_node)
            
            if visited[current_index]:
                continue
            
            visited[current_index] = True
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 四向移动
                neighbor = (current_node[0] + dx, current_node[1] + dy)
                if 0 <= neighbor[0] < n and 0 <= neighbor[1] < n and self.map[neighbor[0], neighbor[1]] == 0:
                    neighbor_index = self.map_to_index(neighbor)
                    new_cost = current_cost + self.resolution
                    
                    if new_cost < dist[neighbor_index]:
                        dist[neighbor_index] = new_cost
                        prev[neighbor_index] = self.map_to_index(current_node)
                        heapq.heappush(priority_queue, (new_cost, neighbor))
        
        # 重建路径
        path = []
        current_index = self.map_to_index(endNode)
        while current_index != -1:
            path.append(self.index_to_map(current_index))
            current_index = prev[current_index]
        path.reverse()
        
        # 如果路径为空，表示没有找到可行路径
        if not path:
            rospy.logwarn("No path found!")
            return []
        
        return path
    
    def map_to_index(self, map_point):
        # 将地图坐标转换为一维数组的索引
        n = self.map.shape[0]
        return map_point[0] * n + map_point[1]
    
    def index_to_map(self, index):
        # 将一维数组的索引转换为地图坐标
        n = self.map.shape[0]
        return (index // n, index % n)
    
    def publish_path(self, path):
        # 将路径发布为ROS Path消息
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0] * self.resolution + self.origin[0]
            pose.pose.position.y = point[1] * self.resolution + self.origin[1]
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        
if __name__ == '__main__':
    planner = DijkstraPlanner()
    rospy.spin()

