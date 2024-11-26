#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np

class Task1(Node):
    """
    Wall-following task to follow the right wall.
    """
    def __init__(self):
        super().__init__('task1_node')
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)        
        self.sub_laser = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        
        self.timer = self.create_timer(0.1, self.timer_cb)
        
        self.state = -1
        self.ang_integral_error=0
        self.ang_previous_error = 0
        self.lin_integral_error=0
        self.lin_previous_error = 0
        self.ang_vel = 0
        self.lin_vel = 0
        self.d = 0.5
        self.f = 1
        self.rd = 0.75

        self.regions = {
            'right': float('inf'),
            'fright': float('inf'),
            'front': float('inf'),
            'fleft': float('inf'),
            'left': float('inf'),
        }

        self.state_dict = {
            0: 'find the wall',
            1: 'turn left',
            2: 'follow the wall',
        }

    def laser_callback(self, msg):
        self.regions = {
            'left':   min(msg.ranges[85:95],  default=10),
            'fleft':  min(msg.ranges[40:50],   default=10),
            'front':  min(msg.ranges[0:1],     default=10),
            'fright': min(msg.ranges[310:320], default=10),
            'right':  min(msg.ranges[265:275], default=10),
        }
        # self.sectors = {
        #     'left':   (msg.ranges[80:100] ),
        #     'fleft':  (msg.ranges[35:55]  ),
        #     'front':  (msg.ranges[0:1]    ),
        #     'fright': (msg.ranges[305:325]),
        #     'right':  (msg.ranges[260:280]),
        # }
        if ((self.regions['front'] < 0.35) or (self.regions['fright'] < 0.25) or (self.regions['fleft'] < 0.25)):
            msg = Twist()
            msg.linear.x = 0.0
            if (self.regions['fleft'] < 0.25):
                msg.angular.z = -0.5
            elif (self.regions['fright'] < 0.25):
                msg.angular.z = 0.5
            self.pub_cmd_vel.publish(msg)
        self.take_action()

    def PID_angular(self, angular_error):
        kp_ang, kd_ang, ki_ang, dt = 25, 3.5, 0.1, 0.1
        self.ang_integral_error += angular_error * dt
        self.ang_integral_error = max(min(self.ang_integral_error, 1), -1)
        ang_derivative = (angular_error - self.ang_previous_error) / dt
        self.ang_previous_error = angular_error
        self.ang_vel = (kp_ang * angular_error) + (ki_ang * self.ang_integral_error) + (kd_ang * ang_derivative)
        self.ang_vel = min(max(self.ang_vel, 0.0001), 0.25)
        return self.ang_vel

    def PID_linear(self, linear_error):
        kp_lin, kd_lin, ki_lin, dt = 55, 5.5, 0.1, 0.1
        self.lin_integral_error += linear_error * dt
        self.lin_integral_error = max(min(self.lin_integral_error, 1.0), -1.0)
        lin_derivative = (linear_error - self.lin_previous_error) / dt
        self.lin_previous_error = linear_error
        self.lin_vel = (kp_lin * linear_error) + (ki_lin * self.lin_integral_error) + (kd_lin * lin_derivative)
        self.lin_vel = min(max(self.lin_vel, 0.0001), 0.35)
        return self.lin_vel
    
    def get_mean_std(self, region_values):
        region_array = np.array(region_values)
        mean_value = np.mean(region_array)
        net_deviation = np.sum((region_array - mean_value))
        return mean_value, net_deviation

    def change_state(self, new_state):
        if self.state != new_state:
            # self.get_logger().info(f"Wall follower - [{new_state}] - {self.state_dict[new_state]}")
            self.state = new_state

    def take_action(self):
        regions = self.regions
        msg = Twist()
        self.pub_cmd_vel.publish(msg)
        
        # for region_name, region_values in self.sectors.items():
        #     mean, std = self.get_mean_std(region_values)
        #     self.get_logger().info(f"Region: {region_name}, Mean: {mean:.2f}, Std Dev: {std:.2f}")

        if regions['front'] > self.f and regions['fleft'] > self.d and regions['fright'] > self.d:
            state_description = 'case 1 - nothing'
            self.change_state(0)
        elif regions['front'] < self.f and regions['fleft'] > self.d and regions['fright'] > self.d:
            state_description = 'case 2 - front'
            self.change_state(1)
        elif regions['front'] > self.d and regions['fleft'] > self.d and regions['fright'] < self.d:
            state_description = 'case 3 - fright'
            self.change_state(2)
        elif regions['front'] > self.d and regions['fleft'] < self.d and regions['fright'] > self.d:
            state_description = 'case 4 - fleft'
            self.change_state(0)
        elif regions['front'] < self.d and regions['fleft'] > self.d and regions['fright'] < self.d:
            state_description = 'case 5 - front and fright'
            self.change_state(1)
        elif regions['front'] < self.d and regions['fleft'] < self.d and regions['fright'] > self.d:
            state_description = 'case 6 - front and fleft'
            self.change_state(1)
        elif regions['front'] < self.d and regions['fleft'] < self.d and regions['fright'] < self.d:
            state_description = 'case 7 - front and fleft and fright'
            self.change_state(1)
        elif regions['front'] > self.d and regions['fleft'] < self.d and regions['fright'] < self.d:
            state_description = 'case 8 - fleft and fright'
            self.change_state(0)
        else:
            state_description = 'unknown case'
            self.change_state(3)
        # self.get_logger().info(f"{state_description}")

    def find_wall(self):
        msg = Twist()
        f = self.regions["front"] > self.f
        d = self.regions["right"] > self.d
        if f:
            msg.linear.x = 0.35
            msg.angular.z = 0.0
        else:
            if d:
                msg.linear.x = 0.0
                msg.angular.z = 0.25
        self.get_logger().info("find wall")
        return msg

    def turn_left(self, err):
        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = self.PID_angular(abs(err)) if err<0 else -self.PID_angular(abs(err))
        self.get_logger().info(f"Leftist - {err} {msg.angular.z}")
        return msg
    
    def turn_right(self, err):
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = - self.PID_angular(abs(err))
        self.get_logger().info(f"Rightist - {err} {msg.angular.z}")
        return msg

    def follow_the_wall(self, err):
        msg = Twist()
        msg.linear.x =  0.3
        msg.angular.z = self.PID_angular(abs(err)) if err<0 else -self.PID_angular(abs(err))
        self.get_logger().info(f"follow  {err} {msg.angular.z}")
        return msg

    def timer_cb(self):
        msg = Twist()
        # if self.state == 0:
        #     msg = self.find_wall()
        # elif self.state == 1:
        #     msg = self.turn_left()
        # elif self.state == 2:
        #     msg = self.follow_the_wall()
        # else:
        #     self.get_logger().error('Unknown state!')
        if (self.regions['front'] > self.f) and (self.regions['fright'] < self.rd) and (self.regions['right'] < self.d):
            msg = self.follow_the_wall((self.regions['right']-self.d))
        elif (self.regions['front'] < self.f) and (self.regions['fright'] < self.rd) and (self.regions['right'] < self.d):
            msg = self.turn_left((self.regions['right']-self.d))
        elif (self.regions['front'] < self.f) and ((self.regions['right'] > 5) or (self.regions['right'] > 5)):
            msg = self.turn_right((self.regions['fright']-self.rd))
        else:
            msg = self.find_wall()
        self.pub_cmd_vel.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    task1 = Task1()
    try:
        rclpy.spin(task1)
    except KeyboardInterrupt:
        pass
    finally:
        task1.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()