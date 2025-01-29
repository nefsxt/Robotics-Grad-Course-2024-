#!/usr/bin env python3

import rospy
import numpy as np
import math
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


class NavFunction():
    def __init__(self, initial_x, initial_y, initial_z, initial_roll, initial_pitch, initial_yaw, goal_x, goal_y, goal_theta, k, k_v, k_a):
        rospy.init_node('my_controller')
        self.pub = rospy.Publisher('/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
        self.scan_subscriber = rospy.Subscriber('/front/scan', LaserScan, self.laser_callback)
        self.odom_subscriber = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.angle_increment = 0.006554075051099062

        self.initial_x = initial_x
        self.initial_y = initial_y
        self.initial_z = initial_z
        self.initial_roll = initial_roll
        self.initial_pitch = initial_pitch
        self.initial_yaw = initial_yaw
        self.initial()

        self.scan_data = []
        self.yaw = 0.0
        self.position = np.zeros(2)
        self.orientation = np.zeros(4)
        self.goal = np.array([goal_x, goal_y])
        self.goal_theta = goal_theta
        
        self.obstacle = False
        self.gamma = 0.0
        self.grad_gamma = np.zeros(2)

        self.beta = 0.0
        self.grad_beta = np.zeros(2)
        self.grad_phi = np.zeros(2)

        self.k = k
        self.k_v = k_v
        self.k_a = k_a


    def set_initial_pose(self):
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            set_model_state = rospy.ServiceProxy(
                '/gazebo/set_model_state', SetModelState)
            state_msg = ModelState()
            state_msg.model_name = 'jackal'
            state_msg.pose.position.x = self.initial_x
            state_msg.pose.position.y = self.initial_y
            state_msg.pose.position.z = self.initial_z
            state_msg.pose.orientation.x = self.initial_roll
            state_msg.pose.orientation.y = self.initial_pitch
            state_msg.pose.orientation.z = self.initial_yaw
            state_msg.pose.orientation.w = 1.0
            set_model_state(state_msg)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def initial(self):
        rospy.loginfo(f'Initial x position = {self.initial_x}')
        rospy.loginfo(f'Initial y position = {self.initial_y}')
        self.set_initial_pose()

    def laser_callback(self, msg):
        self.scan_data = list(msg.ranges)

    def odom_callback(self, data):
        self.position = np.array([data.pose.pose.position.x, data.pose.pose.position.y])
        self.orientation = [data.pose.pose.orientation.x, data.pose.pose.orientation.y,
                            data.pose.pose.orientation.z, data.pose.pose.orientation.w]
        (_, _, self.yaw) = euler_from_quaternion(self.orientation)

    def attractive_potential(self):
        self.gamma = math.sqrt((self.position[0] - self.goal[0])**2 + (self.position[1] - self.goal[1])**2)
        self.grad_gamma = (self.position - self.goal)/self.gamma

    def repulsive_potential(self):
        self.obstacle = False
        step = 30
        safe_zone = 2

        q_i = []
        beta_i = []

        if self.scan_data:
            for i in np.arange(0, 720, step):
                if min(self.scan_data[i:i+step]) <= safe_zone:
                    if not self.obstacle:
                        self.beta = 1.0
                        self.obstacle = True
                    
                    r = min(self.scan_data[i:i+step])
                    dist = r + 0.2 + 0.5
                    self.beta *= dist
                    beta_i.append(dist)
                    angle = (i + np.argmin(self.scan_data[i:i+step])) * self.angle_increment
                    q_i.append(self.position + np.array([r*math.cos(angle), r*math.sin(angle)]))

        if q_i:
            q_i = np.asarray(q_i)
            grad_beta_i = []
            for i in range(len(q_i)):
                grad_beta_i.append(2 * (self.position - q_i[i]))

            self.grad_beta = np.zeros(2)
            for i in range(len(q_i)):
                prod_beta_j = 1.0
                for j in range(len(q_i)):
                    if i != j:
                        prod_beta_j *= beta_i[j]
                self.grad_beta += grad_beta_i[i] * prod_beta_j

    def partial_phi(self):
        if not self.obstacle:
            self.grad_phi = self.grad_gamma
        else:
            a = 1/((self.gamma**(2*self.k) + self.beta)**(2/self.k))
            b = 2 * self.gamma * self.grad_gamma * (self.gamma**(2*self.k) + self.beta)**(1/self.k)
            d = 2 * self.k * self.gamma**(2*self.k - 1) * self.grad_gamma + self.grad_beta
            c = (self.gamma**2) * (1/self.k) * (self.gamma**(2*self.k) + self.beta)**((1-self.k)/self.k) * d
            self.grad_phi = (b - c)*a

    def theta_diff(self):
        self.theta_diff_angular = math.atan2(-1 * self.grad_phi[1], -1 * self.grad_phi[0]) - self.yaw



    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        count = 0
        while not rospy.is_shutdown():
            if count == 0:
                count += 1
                continue

            self.attractive_potential()
            self.repulsive_potential()
            if not self.obstacle:
                self.beta = 0.0
                self.grad_beta = np.zeros(2)
            self.partial_phi()
            self.theta_diff()

            cmd_vel = Twist()
            cmd_vel.linear.x = self.k_v * np.linalg.norm(-1 * self.grad_phi) #
            cmd_vel.angular.z = self.k_a * self.theta_diff_angular  # 

            self.pub.publish(cmd_vel)
            rate.sleep()

            if np.linalg.norm(self.position - self.goal) <= 0.1:
                break




if __name__ == '__main__':
    initial_x = 0.0
    initial_y = 0.0
    initial_yaw = 0.0
    initial_z = initial_roll = initial_pitch = 0.0

    goal_x = 6.5
    goal_y = 4
    goal_theta = 0

    k = 1
    k_v = 0.5
    k_a = 0.5

    nav_func = NavFunction(initial_x, initial_y, initial_z, initial_roll,
                           initial_pitch, initial_yaw, goal_x, goal_y, goal_theta, k, k_v, k_a)
    nav_func.run()
