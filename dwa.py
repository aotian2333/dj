#!/usr/bin/env python
import rospy
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry

class DWAPlanner:
    def __init__(self):
        rospy.init_node('dwa_planner')
        
        # 参数设置
        self.max_speed = 0.5     # 最大线速度 (m/s)
        self.min_speed = 0.1     # 最小线速度
        self.max_yaw_rate = 1.5  # 最大角速度 (rad/s)
        self.max_accel = 0.5     # 最大加速度
        self.robot_radius = 0.3 # 机器人半径
        self.predict_time = 1.5  # 轨迹预测时间
        self.laser_range = 2.0   # 雷达考虑范围
        
        # 订阅者
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # 发布者
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # 状态变量
        self.current_pose = Point()
        self.current_velocity = Twist()
        self.laser_data = None

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose.position
        self.current_velocity = msg.twist.twist

    def laser_callback(self, msg):
        self.laser_data = msg

    def create_dynamic_window(self):
        # 创建动态窗口
        vs = [self.current_velocity.linear.x,
              self.current_velocity.angular.z,
              -self.max_accel,
              self.max_accel,
              -self.max_accel,
              self.max_accel]
        
        return [
            max(self.min_speed, vs[0]-vs[2]*0.1),
            min(self.max_speed, vs[0]+vs[3]*0.1),
            max(-self.max_yaw_rate, vs[1]-vs[4]*0.1),
            min(self.max_yaw_rate, vs[1]+vs[5]*0.1),
        ]

    def motion_predict(self, v, w):
        # 轨迹预测
        dt = 0.1
        x = 0.0
        y = 0.0
        theta = 0.0
        trajectory = []
        
        for _ in range(int(self.predict_time/dt)):
            theta += w * dt
            x += v * math.cos(theta) * dt
            y += v * math.sin(theta) * dt
            trajectory.append((x, y))
            
            if not self.check_collision(x, y):
                return None
                
        return trajectory

    def check_collision(self, x, y):
        # 碰撞检测
        if self.laser_data is None:
            return False
            
        local_x = x - self.current_pose.x
        local_y = y - self.current_pose.y
        dist = math.hypot(local_x, local_y)
        
        if dist > self.laser_range:
            return True
            
        angle = math.atan2(local_y, local_x)
        if angle < self.laser_data.angle_min or angle > self.laser_data.angle_max:
            return True
            
        index = int((angle - self.laser_data.angle_min) / self.laser_data.angle_increment)
        if 0 <= index < len(self.laser_data.ranges):
            range_dist = self.laser_data.ranges[index]
            if range_dist - dist < self.robot_radius:
                return False
                
        return True

    def calculate_scores(self, v, w):
        # 计算轨迹评分
        trajectory = self.motion_predict(v, w)
        if trajectory is None:
            return float('-inf')
            
        # 距离障碍物评分
        min_dist = float('inf')
        for (x, y) in trajectory:
            dx = x - self.current_pose.x
            dy = y - self.current_pose.y
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                
        # 速度评分
        speed_score = v / self.max_speed
        
        # 轨迹方向评分
        heading_score = 1.0 - abs(w)/self.max_yaw_rate
        
        return 0.8*min_dist + 0.1*speed_score + 0.1*heading_score

    def plan(self):
        # 主规划函数
        if self.laser_data is None:
            return
            
        dw = self.create_dynamic_window()
        best_score = float('-inf')
        best_v = 0.0
        best_w = 0.0
        
        # 速度采样
        for v in np.linspace(dw[0], dw[1], 10):
            for w in np.linspace(dw[2], dw[3], 20):
                score = self.calculate_scores(v, w)
                if score > best_score:
                    best_score = score
                    best_v = v
                    best_w = w
                    
        # 生成控制指令
        cmd = Twist()
        cmd.linear.x = best_v
        cmd.angular.z = best_w
        self.cmd_pub.publish(cmd)

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.plan()
            rate.sleep()

if __name__ == '__main__':
    planner = DWAPlanner()
    planner.run()
