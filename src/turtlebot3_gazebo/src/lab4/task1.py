#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Imu
import numpy as np
import time

def euler_from_quaternion(quaternion):
    """Convert quaternion to euler roll, pitch, yaw."""
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class Task1(Node):
    def __init__(self):
        super().__init__('task1_node')
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_laser = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.sub_imu = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        self.rate = self.create_rate(10)

        self.pose_index = 0
        self.preset_yaws = [0, np.pi/2, np.pi, -((np.pi)/2)]
        self.yaw = 0.0
        self.wall_error = 0.0
        self.yaw_error = 100000
        self.cooldown_period = 5
        self.wait = False
        self.last_turn_right_time = 0
        self.front_scan = 0  

        self.state = 0
        self.ang_integral_error=0
        self.ang_previous_error = 0
        self.lin_integral_error=0
        self.lin_previous_error = 0
        self.ang_vel = 0
        self.lin_vel = 0
        self.d = 0.25
        self.f = 0.75
        self.right_open = False

        self.regions = {
            'front' : float('inf'),
            'fleft' : float('inf'),
            'left'  : float('inf'),
            'bleft' : float('inf'),
            'back'  : float('inf'),
            'bright': float('inf'),
            'right' : float('inf'),
            'fright': float('inf')
        }

        self.state_dict = {
            0: 'find the wall',
            1: 'turn left',
            2: 'follow the wall',
        }

    def imu_callback(self, msg):
        orientation_q = msg.orientation
        _, _, yaw = euler_from_quaternion(orientation_q)
        self.yaw = (yaw) if (yaw <= (np.pi)) else (yaw - (2*np.pi))

    def laser_callback(self, msg):
        self.regions = {
            'front' :   min(msg.ranges[0:20]+msg.ranges[340:360], default=10),
            'fleft' :   min(msg.ranges[35:55], default=10),
            'left'  :   min(msg.ranges[80:100], default=10),
            'bleft' :   min(msg.ranges[125:145], default=10),
            'back'  :   min(msg.ranges[170:190], default=10),
            'bright':   min(msg.ranges[215:235], default=10),
            'right' :   min(msg.ranges[260:280], default=10),
            'fright':   min(msg.ranges[305:325], default=10),
        }
        # self.right_open = True if (sum(msg.ranges[269:271])/(len(msg.ranges[269:271]))> 5) else False
        # if time.time() - self.last_turn_right_time > self.cooldown_period:
        self.front_scan = msg.ranges[0]
        # if (self.regions['front'] < 0.2):
        #     msg = Twist()
        #     msg.linear.x = -0.1
        #     self.pub_cmd_vel.publish(msg)
        self.right_open = True if (min(msg.ranges[270:274]) > 2.5) else False


    def follow_the_wall(self):
        """Follows the wall using PID angular control."""
        msg = Twist()
        # self.get_logger().info(f"{self.yaw} {self.yaw_error}")
        # self.get_logger().info(f"! {self.preset_yaws[self.pose_index]}")
        # msg.linear.x = self.PID_linear(abs(self.regions['front']- self.f))
        # msg.linear.x = self.PID_linear(abs(self.regions['front'] - 0.1))
        msg.linear.x = 0.35
        self.yaw_error = (self.preset_yaws[self.pose_index] - self.yaw + np.pi) % (2 * np.pi) - np.pi        
        msg.angular.z = -self.PID_Yaw(abs(self.yaw_error)) if (self.yaw_error < 0.0) else self.PID_Yaw(abs(self.yaw_error))
        # msg.linear.x = 0.0
        # msg.angular.z = 0.0
        self.pub_cmd_vel.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.001)
        if (self.regions['front']<(0.5)):
            self.get_logger().warn("Turn left")
            while (not self.turn_left(1)):
                self.get_logger().warn("turn left")
                rclpy.spin_once(self, timeout_sec=0.01)
                self.stop()
                time.sleep(2000)
                break
            return False
        if (self.right_open==True and self.wait==False):
            self.get_logger().warn("Saw Door...")
            while (not self.turn_right(1, 1.5, 0.35)):
                self.wait == True
                # self.self_timer()
                self.last_turn_right_time = time.time() 
                self.stop(0,5)
                self.right_open = False
                time.sleep(2000)
        return True
    

    def self_timer(self):
        start_time = time.time()
        time.sleep(5)
        self.wait == False


    def find_wall(self):
        msg = Twist()
        # self.get_logger().info(f"? {self.preset_yaws[self.pose_index]}")
        msg.linear.x = self.PID_linear(abs(self.regions['front'] - 0.5))
        self.yaw_error = (self.preset_yaws[self.pose_index] - self.yaw + np.pi) % (2 * np.pi) - np.pi        
        msg.angular.z = -self.PID_Yaw(abs(self.yaw_error)) if (self.yaw_error < 0.0) else self.PID_Yaw(abs(self.yaw_error))
        self.pub_cmd_vel.publish(msg)
        if (self.regions['front'] < 0.5):
            self.stop()
            while (not self.turn_left(1)):
                rclpy.spin_once(self, timeout_sec=0.01)
                self.stop()               
                break
            return True
        return False

    def turn_left(self, i, lin=0.0):
        self.pose_index = (self.pose_index+i) if (self.pose_index<(len(self.preset_yaws)-1)) else ((self.pose_index-len(self.preset_yaws))+i)
        msg = Twist()
        yaw_error = (self.preset_yaws[self.pose_index] - self.yaw + np.pi) % (2 * np.pi) - np.pi        
        while (rclpy.ok()):
            rclpy.spin_once(self, timeout_sec=0.01)
            yaw_error = (self.preset_yaws[self.pose_index] - self.yaw + np.pi) % (2 * np.pi) - np.pi        
            msg.linear.x = lin
            # if (self.regions['fright']<0.15 or self.regions['front']<0.15 or self.regions['fleft']<0.15):
            # msg.linear.x =  self.PID_linear(self.regions['front']-0.2)
            msg.angular.z = -self.PID_Yaw(abs(yaw_error)) if (yaw_error < 0.0) else self.PID_Yaw(abs(yaw_error))
            self.pub_cmd_vel.publish(msg)
            # self.get_logger().info(f"L {msg.linear.x} {self.regions['front']}")
            if (abs(yaw_error)<0.1):
                self.stop()                             
                return True
        return False
    
    def turn_right(self, i, b=1, lin=0.0):
        self.pose_index = (self.pose_index-i) if (self.pose_index>0) else (self.pose_index+len(self.preset_yaws)-i)
        msg = Twist()
        yaw_error = (self.preset_yaws[self.pose_index] - self.yaw + np.pi) % (2 * np.pi) - np.pi        
        while (rclpy.ok()):
            rclpy.spin_once(self, timeout_sec=0.01)
            yaw_error = (self.preset_yaws[self.pose_index] - self.yaw + np.pi) % (2 * np.pi) - np.pi        
            # msg.linear.x = lin  if (lin!=0) else self.PID_linear(self.regions['front']-0.2)
            # if (self.regions['fright']<0.15 or self.regions['front']<0.15 or self.regions['fleft']<0.15):
            #     msg.linear.x = -0.35
            msg.linear.x =  self.PID_linear(self.regions['fright']-0.5)
            msg.angular.z = -self.PID_Yaw(abs(yaw_error)) if (yaw_error < 0.0) else self.PID_Yaw(abs(yaw_error))
            msg.angular.z = b * msg.angular.z
            self.pub_cmd_vel.publish(msg)
            self.get_logger().info(f"R {msg.linear.x}")
            if (abs(yaw_error)<0.1):
                self.stop()               
                return True
        return False


    def PID_Yaw(self, yaw_error):
        """PID controller for yaw correction."""
        kp_ang, kd_ang, ki_ang, dt = 25, 25.5, 0.0001, 0.1
        self.ang_integral_error += yaw_error * dt
        self.ang_integral_error = max(min(self.ang_integral_error, 1), -1)
        ang_derivative = (yaw_error - self.ang_previous_error) / dt
        self.ang_previous_error = yaw_error
        self.ang_vel = (kp_ang * yaw_error) + (ki_ang * self.ang_integral_error) + (kd_ang * ang_derivative)
        self.ang_vel = min(max(self.ang_vel, 0.0001), 0.5)
        return self.ang_vel
    
    def PID_linear(self, linear_error):
        kp_lin, kd_lin, ki_lin, dt = 25, 5.5, 0.001, 0.1
        self.lin_integral_error += linear_error * dt
        self.lin_integral_error = max(min(self.lin_integral_error, 1.0), -1.0)
        lin_derivative = (linear_error - self.lin_previous_error) / dt
        self.lin_previous_error = linear_error
        self.lin_vel = (kp_lin * linear_error) + (ki_lin * self.lin_integral_error) + (kd_lin * lin_derivative)
        self.lin_vel = min(max(self.lin_vel, 0.0001), 0.35)
        return self.lin_vel
    
    def stop(self, ang=0.0, lin=0.0):
        msg = Twist()
        msg.linear.x = ang
        msg.angular.z = lin
        self.pub_cmd_vel.publish(msg)  


    def run(self):
        start_time = time.time()

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.001)             
            elapsed_time = time.time() - start_time

            ### Main timer callback logic ###
            regions = self.regions
            
            if regions['front'] > self.f and regions['right'] > 2.5 and regions['left'] > 1.5:
                # state_description = 'Init Finding wall'
                self.get_logger().warn("Init Finding wall")
                while True:
                    rclpy.spin_once(self, timeout_sec=0.01)
                    if self.find_wall():
                        self.stop()               
                        break
                self.get_logger().warn("found wall")
                # time.sleep(1)



            elif self.front_scan < self.f and regions['right'] < 0.5:
                if (regions['left'] < 0.25):
                    self.get_logger().warn("FL Corner - 180") 
                    if not self.turn_right(2):
                        self.stop()

                self.get_logger().warn("wall approaching - Taking Left")                     
                if not self.turn_left(1):
                    self.stop()
                    # time.sleep(1)
  

            else:
            # try:
                rclpy.spin_once(self, timeout_sec=0.01)
                self.get_logger().warn("following wall") 
                while True:
                    rclpy.spin_once(self, timeout_sec=0.001)
                    if not self.follow_the_wall():
                        self.stop()
                        break
                # self.get_logger().warn(f"{self.regions['left']} {self.regions['right']}")
                    
            # except:
            #     print("oppse")      

            if elapsed_time > 600:
                self.get_logger().error("Time Up")
                break

        self.get_logger().error("byebye")
        self.rate.sleep()



def main(args=None):
    rclpy.init(args=args)
    task1 = Task1()
    try:
        # rclpy.spin(task1)
        task1.run()
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
