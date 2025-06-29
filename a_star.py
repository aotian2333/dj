#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import heapq
import numpy as np

class AStarPlanner:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('a_star_planner', anonymous=True)
        
        # 订阅地图信息
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
            path = self.a_star(self.current_pose, self.goal)
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
    
    def heuristic(self, a, b):
        # 计算两个点之间的曼哈顿距离作为启发式函数
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def a_star(self, start, goal):
        # A*算法
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal:
                break
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 四向移动
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.map.shape[0] and 0 <= neighbor[1] < self.map.shape[1] and self.map[neighbor] == 0:
                    new_cost = cost_so_far[current] + self.resolution
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + self.heuristic(goal, neighbor)
                        heapq.heappush(frontier, (priority, neighbor))
                        came_from[neighbor] = current
            
        path = []
        while current is not None:
            path.append(self.map_to_world(current))
            current = came_from[current]
        path.reverse()
        
        return path
    
    def publish_path(self, path):
        # 将路径发布为ROS Path消息
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = rospy.Time.now()
        for point in path:
            pose = PoseStamped()
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
        
if __name__ == '__main__':
    planner = AStarPlanner()
    rospy.spin()
