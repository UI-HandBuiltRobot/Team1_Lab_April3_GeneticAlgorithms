#!/home/ros_setup/ros2_ws/venv/bin/python

#xarm_controller.py
#-------------

#Description:
#    This module provides the interface with xARM 1s over USB

#Author:
#    Umair Cheema <cheemzgpt@gmail.com>

#Version:
#    1.0.0

#License:
#    Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0)

#Date Created:
#    2024-12-01

#Last Modified:
#    2024-12-13

#Python Version:
#    3.8+

#Usage:
#    Can be imported for controlling xARM 1S robotic arm 
#    Example:
#        from xarm_controller import XARMController

#Dependencies:
#    xarm

import os
import time
import xarm
import numpy as np

class XARMController:

    def __init__(self):
        os.environ['LD_LIBRARY_PATH'] = os.getcwd() 
        self.arm = xarm.Controller('USB')  # NEED TO CHANGE THIS 
        self.end_effector = 0.
        self.state = np.array([0.0,0.0,0.0,0.0,0.0])
        self.get_initial_state()
    
    
    def reset(self):
        self.set_joint_state([0.,-8.,0.,0.,-50.])
        self.move_end_effector(40)
    
    def get_initial_state(self):
        self.state = np.array([self.arm.getPosition(6, degrees=True),
                      self.arm.getPosition(5, degrees=True),
                      self.arm.getPosition(4, degrees=True),
                      self.arm.getPosition(3, degrees=True),
                      self.arm.getPosition(2, degrees=True)
            ])
        self.end_effector = self.arm.getPosition(1, degrees=True)
    
    def update_joints_state(self):
        self.state = np.array([self.arm.getPosition(6, degrees=True),
                      self.arm.getPosition(5, degrees=True),
                      self.arm.getPosition(4, degrees=True),
                      self.arm.getPosition(3, degrees=True),
                      self.arm.getPosition(2, degrees=True)
        ])
    
    def update_end_effector_state(self):
        self.end_effector = self.arm.getPosition(1, degrees=True)


    def get_joints_state(self, radians = False):
        self.update_joints_state()
        state = self.state
        if(radians):
            conversion = np.pi/180
            state = conversion * state
        return state
    
    def set_joint_state(self, position_vector, duration_vector = None, radians=False, wait=True):
        if isinstance(position_vector, list):
            position_vector = np.array(position_vector)
        servos = [6,5,4,3,2]
        q = position_vector.astype(float)
        conversion = 180/np.pi
        if (radians):
            q *= conversion
        
        for i,angle in enumerate(q):
            if (duration_vector):
                self.arm.setPosition(servos[i],q[i], duration_vector[i], wait=wait)
            else:
                self.arm.setPosition(servos[i],q[i],250, wait=wait)

    def move_joint(self, joint_number, angle, duration=250, radians=False, wait=True):
        servos = [6,5,4,3,2]
        angle_val = float(angle)
        if(radians):
            conversion = 180/np.pi
            angle_val *= conversion
        self.arm.setPosition(servos[joint_number], angle_val,duration,wait=wait)

    def get_end_effector_state(self, textual = False):
        self.update_end_effector_state()
        state = self.end_effector
        if(textual):
            if -125 <= state < -90:
                state = 'fully_open'
            elif -90 <= state < -45:
                state = 'partially_open'
            elif -45 <= state < 45:
                state = 'partially_close'
            elif 45 <= state < 90:
                state = 'fully_closed' 
        return state

    def move_end_effector(self, angle):
        self.arm.setPosition(1,float(angle), wait=True)

    def play_waypoints_dense(self, waypoints, playback_hz=20.0, max_step_deg=1.5, final_settle_ms=500, cancel_check=None):
        """Stream dense-resampled full-joint waypoints.

        Args:
            waypoints: Joint targets with shape (N, 5) in degrees ordered as
                [arm6, arm5, arm4, arm3, arm2].
            playback_hz: Streaming frequency in Hz.
            max_step_deg: Maximum allowed per-joint delta between streamed commands.
            final_settle_ms: Blocking settle duration for final command.
            cancel_check: Optional callable returning True to cancel playback.

        Returns:
            bool: True on successful completion, False on cancel or command failure.
        """
        try:
            trajectory = np.asarray(waypoints, dtype=float)
        except Exception:
            return False

        if trajectory.ndim != 2 or trajectory.shape[1] != 5:
            return False
        if trajectory.shape[0] == 0:
            return True
        if not np.isfinite(trajectory).all():
            return False

        playback_hz = float(playback_hz)
        max_step_deg = float(max_step_deg)
        final_settle_ms = int(final_settle_ms)
        if playback_hz <= 0.0 or max_step_deg <= 0.0:
            return False

        max_step_deg = max(0.1, max_step_deg)
        final_settle_ms = max(1, final_settle_ms)

        current = np.asarray(self.get_joints_state(radians=False), dtype=float).reshape(5)
        dense_points = []
        previous = current.copy()

        for target in trajectory:
            target = np.asarray(target, dtype=float).reshape(5)
            delta = target - previous
            max_delta = float(np.max(np.abs(delta)))
            steps = max(1, int(np.ceil(max_delta / max_step_deg)))

            for alpha in np.linspace(0.0, 1.0, steps + 1, endpoint=True)[1:]:
                dense_points.append(previous + (delta * alpha))

            previous = target

        if len(dense_points) == 0:
            return True

        interval = 1.0 / playback_hz
        step_duration_ms = max(1, int(round(interval * 1000.0)))
        servos = [6, 5, 4, 3, 2]
        stream_start = time.time()

        try:
            for index, target in enumerate(dense_points):
                if callable(cancel_check) and bool(cancel_check()):
                    return False

                target_time = stream_start + (index * interval)
                sleep_time = target_time - time.time()
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

                full_cmd = [[servo_id, float(target[servo_index])] for servo_index, servo_id in enumerate(servos)]
                self.arm.setPosition(full_cmd, duration=step_duration_ms, wait=False)

            final_target = np.asarray(dense_points[-1], dtype=float).reshape(5)
            settle_duration = max(final_settle_ms, 2 * step_duration_ms)
            final_cmd = [[servo_id, float(final_target[servo_index])] for servo_index, servo_id in enumerate(servos)]
            self.arm.setPosition(final_cmd, duration=settle_duration, wait=True)
        except Exception:
            return False

        return True
    