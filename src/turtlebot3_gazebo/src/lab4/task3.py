#!/usr/bin/env python3

import sys
import os
import math
import yaml
import math
import heapq

import rclpy
import rclpy.logging
from rclpy.node import Node
import rclpy.time

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Pose, Twist, PointStamped
from nav_msgs.msg import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import convolve
from PIL import Image, ImageOps
from graphviz import Graph
from copy import copy, deepcopy
from collections import defaultdict


## CHANGE MAP LOCATION
map_name = '/home/lucifer/sim_ws/src/turtlebot3_gazebo/maps/map'
# map_name = '/home/lucifer/sim_ws/src/turtlebot3_gazebo/maps/map'


## CLASS FOR MAP
class Map():
    def __init__(self, map_name):
        self.map_im, self.map_df, self.limits = self.__open_map(map_name)
        self.image_array = self.__get_obstacle_map(self.map_im, self.map_df)

    def __repr__(self):
        fig, ax = plt.subplots(dpi=150)
        ax.imshow(self.image_array,extent=self.limits, cmap=cm.gray)
        ax.plot()
        return ""

    def __open_map(self,map_name):
        # Open the YAML file which contains the map name and other
        # configuration parameters
        f = open(map_name + '.yaml', 'r')
        map_df = pd.json_normalize(yaml.safe_load(f))
        # Open the map image
        # map_name = map_df.image[0]
        im = Image.open(map_name+'.pgm')
        size = 200, 200
        im.thumbnail(size)
        im = ImageOps.grayscale(im)
        # Get the limits of the map. This will help to display the map
        # with the correct axis ticks.
        xmin = map_df.origin[0][0]
        xmax = map_df.origin[0][0] + im.size[0] * map_df.resolution[0]
        ymin = map_df.origin[0][1]
        ymax = map_df.origin[0][1] + im.size[1] * map_df.resolution[0]

        return im, map_df, [xmin,xmax,ymin,ymax]

    def __get_obstacle_map(self,map_im, map_df):
        img_array = np.reshape(list(self.map_im.getdata()),(self.map_im.size[1],self.map_im.size[0]))
        up_thresh = self.map_df.occupied_thresh[0]*255
        low_thresh = self.map_df.free_thresh[0]*255

        for j in range(self.map_im.size[0]):
            for i in range(self.map_im.size[1]):
                if img_array[i,j] > up_thresh:
                    img_array[i,j] = 255
                else:
                    img_array[i,j] = 0
        return img_array


## CLASS FOR QUEUE
class Queue():
    def __init__(self, init_queue = []):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if(i == self.start):
                tmpstr += "<"
                flag = True
            if(i == self.end):
                tmpstr += ">"
                flag = True

            if(flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self,init_queue = []):
        self.queue = copy(init_queue)

    def sort(self,key=str.lower):
        self.queue = sorted(self.queue,key=key)

    def push(self,data):
        self.queue.append(data)
        self.end += 1

    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue)-1
        return p


# CLASS FOR TREE
class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}
        self.g_visual = Graph('G')

    def __call__(self):
        for name,node in self.g.items():
            if(self.root == name):
                self.g_visual.node(name,name,color='red')
            elif(self.end == name):
                self.g_visual.node(name,name,color='blue')
            else:
                self.g_visual.node(name,name)
            for i in range(len(node.children)):
                c = node.children[i]
                w = node.weight[i]
                #print('%s -> %s'%(name,c.name))
                if w == 0:
                    self.g_visual.edge(name,c.name)
                else:
                    self.g_visual.edge(name,c.name,label=str(w))
        return self.g_visual

    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        # These are exclusive conditions
        self.root = True
        self.end = False

    def set_as_end(self,node):
        # These are exclusive conditions
        self.root = False
        self.end = True


class Nodes():
    def __init__(self,name):
        self.name = name
        self.children = []
        self.weight = []

    def __repr__(self):
        return self.name

    def add_children(self,node,w=None):
        if w == None:
            w = [1]*len(node)
        self.children.extend(node)
        self.weight.extend(w)


# ## CLASS TO PROCESS MAP
class MapProcessor():
    def __init__(self,name):
        self.map = Map(name)
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        self.map_graph = Tree(name)

    def __modify_map_pixel(self,map_array,i,j,value,absolute):
        if( (i >= 0) and
            (i < map_array.shape[0]) and
            (j >= 0) and
            (j < map_array.shape[1]) ):
            if absolute:
                map_array[i][j] = value
            else:
                map_array[i][j] += value

    def __inflate_obstacle(self,kernel,map_array,i,j,absolute):
        dx = int(kernel.shape[0]//2)
        dy = int(kernel.shape[1]//2)
        if (dx == 0) and (dy == 0):
            self.__modify_map_pixel(map_array,i,j,kernel[0][0],absolute)
        else:
            for k in range(i-dx,i+dx):
                for l in range(j-dy,j+dy):
                    self.__modify_map_pixel(map_array,k,l,kernel[k-i+dx][l-j+dy],absolute)

    def inflate_map(self,kernel,absolute=True):
        # Perform an operation like dilation, such that the small wall found during the mapping process
        # are increased in size, thus forcing a safer path.
        self.inf_map_img_array = np.zeros(self.map.image_array.shape)
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.map.image_array[i][j] == 0:
                    self.__inflate_obstacle(kernel,self.inf_map_img_array,i,j,absolute)
        r = np.max(self.inf_map_img_array)-np.min(self.inf_map_img_array)
        if r == 0:
            r = 1
        self.inf_map_img_array = (self.inf_map_img_array - np.min(self.inf_map_img_array))/r

    def get_graph_from_map(self):
        # Create the nodes that will be part of the graph, considering only valid nodes or the free space
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    node = Nodes('%d,%d'%(i,j))
                    self.map_graph.add_node(node)
        # Connect the nodes through edges
        for i in range(self.map.image_array.shape[0]):
            for j in range(self.map.image_array.shape[1]):
                if self.inf_map_img_array[i][j] == 0:
                    if (i > 0):
                        if self.inf_map_img_array[i-1][j] == 0:
                            # add an edge up
                            child_up = self.map_graph.g['%d,%d'%(i-1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up],[1])
                    if (i < (self.map.image_array.shape[0] - 1)):
                        if self.inf_map_img_array[i+1][j] == 0:
                            # add an edge down
                            child_dw = self.map_graph.g['%d,%d'%(i+1,j)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw],[1])
                    if (j > 0):
                        if self.inf_map_img_array[i][j-1] == 0:
                            # add an edge to the left
                            child_lf = self.map_graph.g['%d,%d'%(i,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_lf],[1])
                    if (j < (self.map.image_array.shape[1] - 1)):
                        if self.inf_map_img_array[i][j+1] == 0:
                            # add an edge to the right
                            child_rg = self.map_graph.g['%d,%d'%(i,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_rg],[1])
                    if ((i > 0) and (j > 0)):
                        if self.inf_map_img_array[i-1][j-1] == 0:
                            # add an edge up-left
                            child_up_lf = self.map_graph.g['%d,%d'%(i-1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_lf],[np.sqrt(2)])
                    if ((i > 0) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i-1][j+1] == 0:
                            # add an edge up-right
                            child_up_rg = self.map_graph.g['%d,%d'%(i-1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_up_rg],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j > 0)):
                        if self.inf_map_img_array[i+1][j-1] == 0:
                            # add an edge down-left
                            child_dw_lf = self.map_graph.g['%d,%d'%(i+1,j-1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_lf],[np.sqrt(2)])
                    if ((i < (self.map.image_array.shape[0] - 1)) and (j < (self.map.image_array.shape[1] - 1))):
                        if self.inf_map_img_array[i+1][j+1] == 0:
                            # add an edge down-right
                            child_dw_rg = self.map_graph.g['%d,%d'%(i+1,j+1)]
                            self.map_graph.g['%d,%d'%(i,j)].add_children([child_dw_rg],[np.sqrt(2)])

    def gaussian_kernel(self, size, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        r = np.max(g)-np.min(g)
        sm = (g - np.min(g))*1/r
        return sm

    def rect_kernel(self, size, value):
        m = np.ones(shape=(size,size))
        return m

    def draw_path(self,path):
        path_tuple_list = []
        path_array = copy(self.inf_map_img_array)
        for idx in path:
            tup = tuple(map(int, idx.split(',')))
            path_tuple_list.append(tup)
            path_array[tup] = 0.5
        return path


## CLASS FOR A STAR
# class AStar():
#     def __init__(self,in_tree):
#         self.in_tree = in_tree
#         self.q = Queue()
#         self.dist = {name:np.Inf for name,node in in_tree.g.items()}
#         self.h = {name:0 for name,node in in_tree.g.items()}

#         for name,node in in_tree.g.items():
#             start = tuple(map(int, name.split(',')))
#             end = tuple(map(int, self.in_tree.end.split(',')))
#             self.h[name] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)

#         self.via = {name:0 for name,node in in_tree.g.items()}
#         for __,node in in_tree.g.items():
#             self.q.push(node)

#     def __get_f_score(self,node):
#         idx = node.name
#         return self.dist[idx] + self.h[idx]

#     def solve(self, sn, en):
#       self.dist[sn.name] = 0
#       self.q.push(sn)

#       while len(self.q) > 0:
#           current_node = min(self.q.queue, key=lambda n: self.__get_f_score(n))
#           self.q.queue.remove(current_node)

#           if current_node.name == en.name:
#               break

#           for i in range(len(current_node.children)):
#               child = current_node.children[i]
#               weight = current_node.weight[i]

#               tentative_g_score = self.dist[current_node.name] + weight

#               if tentative_g_score < self.dist[child.name]:
#                   self.dist[child.name] = tentative_g_score
#                   self.via[child.name] = current_node.name

#                   if child not in self.q.queue:
#                       self.q.push(child)

#     def reconstruct_path(self,sn,en):
#         path = []
#         node = en.name
#         dist = self.dist[node]
#         node = en.name
#         while node != sn.name:
#           path.append(node)
#           node = self.via[node]

#         path.append(sn.name)
#         path.reverse()
#         return path,dist


class AStar():
    def __init__(self, in_tree):
        self.in_tree = in_tree
        self.open_set = []  # Priority queue for the open set
        self.dist = {name: np.inf for name, node in in_tree.g.items()}  # g-score
        self.h = {name: 0 for name, node in in_tree.g.items()}  # heuristic
        self.via = {name: None for name, node in in_tree.g.items()}  # Path reconstruction helper

        # Precompute heuristic values (Euclidean distance to the end node)
        for name, node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

    def __get_f_score(self, node_name):
        # f(n) = g(n) + h(n)
        return self.dist[node_name] + self.h[node_name]

    def solve(self, sn, en):
        # Initialize starting node
        self.dist[sn.name] = 0
        heapq.heappush(self.open_set, (self.__get_f_score(sn.name), sn.name))  # Use node name instead of the object

        while self.open_set:
            _, current_node_name = heapq.heappop(self.open_set)

            # If we reached the goal, stop searching
            if current_node_name == en.name:
                break

            current_node = self.in_tree.g[current_node_name]  # Get the actual node object

            for i in range(len(current_node.children)):
                child = current_node.children[i]
                weight = current_node.weight[i]
                tentative_g_score = self.dist[current_node.name] + weight

                if tentative_g_score < self.dist[child.name]:
                    # Update distance and via node
                    self.dist[child.name] = tentative_g_score
                    self.via[child.name] = current_node.name

                    # Push to the open set with updated f-score
                    heapq.heappush(self.open_set, (self.__get_f_score(child.name), child.name))  # Use node name


    def reconstruct_path(self, sn, en):
        # Backtrack from the goal to the start to reconstruct the path
        path = []
        node_name = en.name
        total_dist = self.dist[node_name]

        while node_name is not None:
            path.append(node_name)
            node_name = self.via[node_name]

        path.reverse()
        return path, total_dist




class Task3(Node):

    def __init__(self, node_name='Navigation'):

        super().__init__(node_name)
        self.path = Path()
        self.goal_pose = PoseStamped()
        self.ttbot_pose = PoseStamped()
        self.goal_point = PointStamped()

        self.create_subscription(PoseStamped, '/move_base_simple/goal', self.__goal_pose_cbk, 10)
        # self.create_subscription(PointStamped, '/robot/clicked_point', self.__goal_point_cbk, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.__ttbot_pose_cbk, 10)

        self.path_pub = self.create_publisher(Path, '/plan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.rate = self.create_rate(10)

        self.speed = 0
        self.heading = 0
        self.lin_integral_error = 0
        self.lin_previous_error = 0
        self.ang_integral_error = 0
        self.ang_previous_error = 0
        self.wp_reached = False
        self.index = 0
        self.prev_idx = 0

        # self.world_dim = [3 ,2]
        # self.graph_dim = [42, 60]
        self.world_dim = [10 ,7]
        self.graph_dim = [200, 140]
        self.world_origin = [-1.75, -2]
        # self.world_origin = [-1.5, -0.5]
        self.graph_origin = [75, 55]
        # self.graph_origin = [10, 35]

        self.scale = [13,14]
        self.res   = [20, 20] 


    def __goal_point_cbk(self, data):
        self.goal_point = data

    def __goal_pose_cbk(self, data):
        self.goal_pose = data

    def __ttbot_pose_cbk(self, data):
        self.ttbot_pose = data.pose


    def graph2pose(self, graph_y, graph_x):
        dx = graph_x - self.graph_origin[0]
        dy = graph_y - self.graph_origin[1]
        world_x = ( dx  / self.res[0]) 
        world_y = (-dy  / self.res[1]) 
        world_x = world_x + self.world_origin[0]
        world_y = world_y + self.world_origin[1]    
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position.x = world_x
        pose_stamped.pose.position.y = world_y
        pose_stamped.pose.position.z = 0.0  
        pose_stamped.pose.orientation.w = 1.0  
        return pose_stamped
    
    def point2pose(self, data):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position.x = data.point.x
        pose_stamped.pose.position.y = data.point.y
        pose_stamped.pose.position.z = 0.0  
        pose_stamped.pose.orientation.w = 1.0  
        return pose_stamped

    def pose2graph(self, pose):
        world_x = pose.pose.position.x
        world_y = pose.pose.position.y
        dx = world_x 
        dy = world_y 
        graph_x = int( dx * self.scale[0])
        graph_y = int(-dy * self.scale[1])
        graph_x = graph_x + self.graph_origin[0]
        graph_y = graph_y + self.graph_origin[1]
        node_name = f"{graph_y},{graph_x}"
        return node_name



    def get_yaw_from_pose(self, pose):
        orientation_q = pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y ** 2 + orientation_q.z ** 2)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def PID_angular(self, angular_error):
        kp_ang, kd_ang, ki_ang, dt = 35, 35.5, 0.0001, 0.1
        self.ang_integral_error += angular_error * dt
        self.ang_integral_error = max(min(self.ang_integral_error, 1), -1)  # Anti-windup
        ang_derivative = (angular_error - self.ang_previous_error) / dt
        self.ang_previous_error = angular_error
        self.ang_vel = (kp_ang * angular_error) + (ki_ang * self.ang_integral_error) + (kd_ang * ang_derivative)
        self.ang_vel = min(max(abs(self.ang_vel), 0.0), 0.35)
        return self.ang_vel

    def PID_linear(self, linear_error):
        kp_lin, kd_lin, ki_lin, dt = 5.0, 2.5, 0.001, 0.1
        self.lin_integral_error += linear_error * dt
        self.lin_integral_error = max(min(self.lin_integral_error, 1.0), -1.0)  # Anti-windup
        lin_derivative = (linear_error - self.lin_previous_error) / dt
        self.lin_previous_error = linear_error
        self.lin_vel = (kp_lin * linear_error) + (ki_lin * self.lin_integral_error) + (kd_lin * lin_derivative)
        self.lin_vel = min(max(self.lin_vel, 0.0), 0.35)
        return self.lin_vel
    
    def reached_goal(self, current_pose, target_pose, tolerance=0.15):
        dx = target_pose.pose.position.x - current_pose.pose.position.x
        dy = target_pose.pose.position.y - current_pose.pose.position.y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        # print(distance < tolerance)
        return distance < tolerance



    def a_star_path_planner(self, start_pose, end_pose):
        path = Path()
        start_node_name = self.pose2graph(start_pose)
        end_node_name = self.pose2graph(end_pose)

        mp = MapProcessor(map_name)
        self.map = mp
        kr = mp.rect_kernel(11,1)
        mp.inflate_map(kr,True)
        mp.get_graph_from_map()
        mp.map_graph.root = start_node_name
        mp.map_graph.end = end_node_name

        if  ((mp.map_graph.root=="") or (mp.map_graph.end=="")):
            self.get_logger().error("Start or End node not found in map.")
            return path

        a_star = AStar(mp.map_graph)
        a_star.solve(mp.map_graph.g[mp.map_graph.root],mp.map_graph.g[mp.map_graph.end])
        node_path, _ = a_star.reconstruct_path(mp.map_graph.g[mp.map_graph.root],mp.map_graph.g[mp.map_graph.end])

        for node_name in node_path:
            graph_x, graph_y = map(int, node_name.split(','))
            waypoint = PoseStamped()
            waypoint = self.graph2pose(graph_x, graph_y)
            path.poses.append(waypoint)
        return node_path, path



    def get_path_idx(self, vehicle_pose, path, prev_idx):
        min_distance = np.inf
        angle_threshold = 0.05
        closest_deviation_idx = None
        vehicle_x = vehicle_pose.pose.position.x
        vehicle_y = vehicle_pose.pose.position.y
        vehicle_yaw = self.get_yaw_from_pose(vehicle_pose)
        
        for i in range((prev_idx+1), len(path.poses)):
            waypoint = path.poses[i]
            waypoint_x = ((waypoint.pose.position.x)) 
            waypoint_y = ((waypoint.pose.position.y))
            distance = np.sqrt((vehicle_x - waypoint_x) ** 2 + (vehicle_y - waypoint_y) ** 2)
            target_angle = math.atan2(waypoint_y - vehicle_y, waypoint_x - vehicle_x)
            angle_deviation = abs((target_angle - vehicle_yaw))
            
            if angle_deviation > angle_threshold and distance < min_distance:
                min_distance = distance
                closest_deviation_idx = i
        
        if closest_deviation_idx is not None:
            return closest_deviation_idx
        
        return (len(path.poses) - 1)



    def path_follower(self, vehicle_pose, current_goal, prev_goal):
        linear_error_margin = 0.35
        angular_error_margin = 0.12

        vehicle_x = vehicle_pose.pose.position.x
        vehicle_y = vehicle_pose.pose.position.y
        
        vehicle_yaw = self.get_yaw_from_pose(vehicle_pose)

        goal_x = (current_goal.pose.position.x)
        goal_y = (current_goal.pose.position.y)

        prev_goal_x = (prev_goal.pose.position.x) if (self.prev_idx>0) else 0
        prev_goal_y = (prev_goal.pose.position.y) if (self.prev_idx>0) else 0

        lin_ex = goal_x - vehicle_x
        lin_ey = goal_y - vehicle_y

        ang_ex = goal_x - prev_goal_x
        ang_ey = goal_y - prev_goal_y


        distance_to_goal = math.sqrt((lin_ex) ** 2 + (lin_ey) ** 2)
        target_angle = (math.atan2(ang_ey, ang_ex))
        angle_diff = self.normalize_angle(target_angle - vehicle_yaw)

        if ((abs(lin_ex)<linear_error_margin)) and ((abs(lin_ey)<linear_error_margin)) and (abs(angle_diff)<angular_error_margin):
            self.wp_reached = True
            self.get_logger().info('waypoint reached')
            speed = self.PID_linear(distance_to_goal)
            heading = self.PID_angular(abs(angle_diff)) if angle_diff > 0 else -self.PID_angular(abs(angle_diff))
        else: 
            self.wp_reached = False
            # self.get_logger().warn(f" G {self.index} {round(target_angle, 2)} {round(goal_x, 2)} {round(goal_y, 2)}")
            if (abs(angle_diff) > angular_error_margin):
                speed = 0.05 * distance_to_goal
                heading = self.PID_angular(abs(angle_diff)) if angle_diff > 0 else -self.PID_angular(abs(angle_diff))
                self.get_logger().warn(f" A {self.prev_idx} {self.index} {round(angle_diff, 2)} {round(distance_to_goal, 2)} ")
            else:
                speed = self.PID_linear(distance_to_goal)
                heading = 0.05 * angle_diff 
                # heading = self.PID_angular(abs(angle_diff)) if angle_diff > 0 else -self.PID_angular(abs(angle_diff))
                self.get_logger().warn(f" L {self.prev_idx} {self.index} {round(angle_diff, 2)} {round(distance_to_goal, 2)} ")           

        return speed, heading


    def path_tracer(self, current_goal, prev_goal):
        goal_x = (current_goal.pose.position.x)
        goal_y = (current_goal.pose.position.y)
        prev_goal_x = (prev_goal.pose.position.x)
        prev_goal_y = (prev_goal.pose.position.y)
        wp_ex = abs(goal_x - prev_goal_x)
        wp_ey = abs(goal_y - prev_goal_y)
        distance = math.sqrt((wp_ex) ** 2 + (wp_ey) ** 2)
        target_angle = self.normalize_angle(math.atan2(wp_ey, wp_ex)) 
        self.get_logger().warn(f" c {self.index} {round(target_angle, 2)} {round(goal_x, 2)} {round(goal_y, 2)}")          
        self.get_logger().warn(f" p {self.prev_idx} {round(target_angle, 2)} {round(prev_goal_x, 2)} {round(prev_goal_y, 2)}")           



    def move_ttbot(self, speed, heading):
        cmd_vel = Twist()
        cmd_vel.linear.x = float(speed)
        cmd_vel.angular.z = float(heading)
        # cmd_vel.linear.x = 0.0
        # cmd_vel.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd_vel)

    def publish_path(self, poses):
        path_arr = Path()
        path_arr = poses
        path_arr.header.frame_id = "/map"
        offset_x = 2.75
        offset_y = 3
        for pose in path_arr.poses:
            pose.pose.position.x = 1.5 * (pose.pose.position.x) + offset_x
            pose.pose.position.y = 1.5 * (pose.pose.position.y) + offset_y
        self.path_pub.publish(path_arr)

    def display_path(self, path, node_path):
        path_arr_as = self.map.draw_path(node_path)
        path_arr_as = path_arr_as.astype(float)
        x_coords = [pose.pose.position.x for pose in path.poses]
        y_coords = [pose.pose.position.y for pose in path.poses]
        mp = Map(map_name)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=150)
        axes[0].imshow(mp.image_array, extent=mp.limits, cmap=cm.gray)
        axes[0].plot(x_coords, y_coords, color='blue', linewidth=2, marker='o', markersize=3, label="Path")
        axes[0].legend()
        axes[0].set_title("Map with Path")
        im = axes[1].imshow(path_arr_as, cmap='viridis')  
        fig.colorbar(im, ax=axes[1], orientation="vertical")
        axes[1].set_title("Path Array")
        plt.show()



    def run(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if (self.goal_pose.pose.position.x != self.ttbot_pose.pose.position.x or self.goal_pose.pose.position.y != self.ttbot_pose.pose.position.y):
                self.get_logger().info('Planning path \n> from {}, {} \n> to {}, {}'.format(self.ttbot_pose.pose.position.x, self.ttbot_pose.pose.position.y, self.goal_pose.pose.position.x, self.goal_pose.pose.position.y))
                node_path, path = self.a_star_path_planner(self.ttbot_pose, self.goal_pose)
                self.publish_path(path)
                # self.display_path(path, node_path)
                prev_idx = -1
                while not self.reached_goal(self.ttbot_pose, self.goal_pose):
                    rclpy.spin_once(self, timeout_sec=0.1)
                    idx = self.get_path_idx(self.ttbot_pose, path, prev_idx)
                    self.index = idx
                    self.prev_idx = prev_idx
                    
                    if idx != -1:
                        current_goal = path.poses[idx]
                        prev_goal = path.poses[prev_idx]
                        self.get_logger().info(f'ttx: {self.ttbot_pose.pose.position.x}, tty: {self.ttbot_pose.pose.position.y}')
                        while (self.wp_reached != True):
                            rclpy.spin_once(self, timeout_sec=0.1)
                            speed, heading = self.path_follower(self.ttbot_pose, current_goal, prev_goal)
                            self.move_ttbot(speed, heading)
                    
                    
                    # while (idx<=len(node_path)):
                    #     self.index = idx
                    #     self.prev_idx = idx-1                       
                    #     current_goal = path.poses[idx]
                    #     prev_goal = path.poses[idx-1]
                    #     self.path_tracer(current_goal, prev_goal)
                    #     idx = idx + 1
                    
                    
                    print ("------------------------- wp reached ------------------------------")
                    prev_idx = idx 
                    self.wp_reached = False

                # Once all waypoints are reached, stop the robot
                self.get_logger().info("Goal reached, stopping robot")
                self.move_ttbot(0.0, 0.0)
                break
        self.rate.sleep()


def main(args=None):
    rclpy.init(args=args)

    task3 = Task3()

    try:
        rclpy.spin(task3)
    except KeyboardInterrupt:
        pass
    finally:
        task3.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
