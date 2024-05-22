import math
import numpy as np

def calculate_racing_line(waypoints):
    racing_line = []
    for i in range(len(waypoints)):
        if i == 0 or i == len(waypoints) - 1:
            # Start and end points stay the same
            racing_line.append(waypoints[i])
        else:
            prev_point = waypoints[i - 1]
            current_point = waypoints[i]
            next_point = waypoints[i + 1]
            
            # Calculate direction vectors
            vector_a = np.array(current_point) - np.array(prev_point)
            vector_b = np.array(next_point) - np.array(current_point)
            
            # Calculate angles
            angle = np.arctan2(vector_b[1], vector_b[0]) - np.arctan2(vector_a[1], vector_a[0])
            angle = np.degrees(angle)
            
            # Adjust point slightly towards the inside of the curve
            if angle > 0:
                adjusted_point = (
                    current_point[0] - 0.1 * (current_point[0] - prev_point[0]),
                    current_point[1] - 0.1 * (current_point[1] - prev_point[1])
                )
            else:
                adjusted_point = (
                    current_point[0] + 0.1 * (next_point[0] - current_point[0]),
                    current_point[1] + 0.1 * (next_point[1] - current_point[1])
                )
            racing_line.append(adjusted_point)
    
    return racing_line

def reward_function(params):
    # Define the waypoints for the track
    waypoints = params['waypoints']
    closest_waypoints = params['closest_waypoints']
    
    # Calculate the racing line based on waypoints
    racing_line = calculate_racing_line(waypoints)
    
    # Extract necessary parameters from 'params'
    x = params['x']
    y = params['y']
    distance_from_center = params['distance_from_center']
    track_width = params['track_width']
    heading = params['heading']
    speed = params['speed']
    steering_angle = params['steering_angle']
    all_wheels_on_track = params['all_wheels_on_track']
    
    # Find the closest point on the racing line
    min_dist = float('inf')
    closest_racing_point = None
    for point in racing_line:
        dist = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_racing_point = point
    
    # Calculate reward for following the racing line
    distance_reward = 1.0 - min_dist / (track_width / 2)
    
    # Reward for maintaining a higher speed
    SPEED_THRESHOLD = 3.0  # Increased speed threshold to encourage faster driving
    if speed > SPEED_THRESHOLD:
        speed_reward = 2.0  # Higher reward for exceeding the speed threshold
    else:
        speed_reward = speed / SPEED_THRESHOLD  # Proportional reward up to the threshold
    
    # Penalize for excessive steering to prevent zig-zag behavior
    STEERING_THRESHOLD = 20.0  # Allow a bit more steering
    if abs(steering_angle) > STEERING_THRESHOLD:
        steering_penalty = 0.7  # Increased penalty for excessive steering
    else:
        steering_penalty = 1.0
    
    # Ensure all wheels are on track to get a decent reward
    if not all_wheels_on_track:
        reward = 1e-3
    else:
        # Combine the rewards and penalties
        reward = distance_reward * speed_reward * steering_penalty
    
    return float(reward)
