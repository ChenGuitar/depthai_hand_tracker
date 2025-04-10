import cv2
import numpy as np

JOINT_NAMES = [
    # 拇指 (索引1-3)
    ("Thumb", "MCP"),  # 近端关节
    ("Thumb", "PIP"),  # 中间关节
    ("Thumb", "DIP"),  # 远端关节
    
    # 食指 (索引5-8)
    ("Index", "MCP"),
    ("Index", "PIP"), 
    ("Index", "DIP"),
    
    # 中指 (索引9-12)
    ("Middle", "MCP"),
    ("Middle", "PIP"),
    ("Middle", "DIP"),
    
    # 无名指 (索引13-16)
    ("Ring", "MCP"),
    ("Ring", "PIP"),
    ("Ring", "DIP"),
    
    # 小指 (索引17-20)
    ("Pinky", "MCP"),
    ("Pinky", "PIP"),
    ("Pinky", "DIP"),
]


def compute_axis_angle_params(points):
    # Define the joints for each finger (excluding wrist and fingertips)
    fingers = [
        [1, 2, 3],    # Thumb (joints 1, 2, 3)
        [5, 6, 7],    # Index finger
        [9, 10, 11],  # Middle finger
        [13, 14, 15], # Ring finger
        [17, 18, 19]  # Pinky finger
    ]
    
    # 1. Build wrist coordinate system
    wrist_pos = points[0]
    index_root_pos = points[5]
    pinky_root_pos = points[17]
    
    v1 = index_root_pos - wrist_pos
    v2 = pinky_root_pos - wrist_pos
    n = np.cross(v1, v2)
    n_normalized = n / (np.linalg.norm(n))
    
    # Wrist coordinate system
    x_axis_wrist = v1 / (np.linalg.norm(v1))
    y_axis_wrist = np.cross(x_axis_wrist, n_normalized)
    y_axis_wrist /= np.linalg.norm(y_axis_wrist)
    z_axis_wrist = np.cross(x_axis_wrist, y_axis_wrist)
    z_axis_wrist /= np.linalg.norm(z_axis_wrist)
    C_wrist = (x_axis_wrist, y_axis_wrist, z_axis_wrist)
    
    coordinate_systems = {}
    
    # 2. Build coordinate systems for each joint
    # 坐标系约定（右手系）：
    # - X轴：沿骨骼指向远端（屈曲方向）
    # - Y轴：垂直骨骼平面（外展方向）
    # - Z轴：骨骼轴向旋转方向

    # 角度正方向定义：
    # - Flexion：正向弯曲（握拳方向）
    # - Abduction：手指展开方向
    # - Rotation：拇指为内旋方向，其他手指为外旋方向
    for finger in fingers:
        for i, joint in enumerate(finger):
            current_pos = points[joint]
            
            # Determine next joint (skip fingertips)
            next_joint = finger[i+1] if i < len(finger)-1 else None
            if next_joint is None:
                next_joint = joint + 1  # Fallback, adjust according to your data
            
            # X-axis points to next joint
            x_axis = points[next_joint] - current_pos
            x_axis_normalized = x_axis / (np.linalg.norm(x_axis))
            
            # Root joints use palm normal, others use parent's coordinate system
            if i == 0:
                # Root joint (parent is wrist)
                y_axis = np.cross(x_axis_normalized, n_normalized)
            else:
                # Non-root joint (parent is previous joint)
                parent_joint = finger[i-1]
                parent_pos = points[parent_joint]
                parent_vec = current_pos - parent_pos
                parent_vec_normalized = parent_vec / (np.linalg.norm(parent_vec))
                y_axis = np.cross(x_axis_normalized, parent_vec_normalized)
            
            # Orthonormalize axes
            y_axis_normalized = y_axis / (np.linalg.norm(y_axis))
            z_axis = np.cross(x_axis_normalized, y_axis_normalized)
            z_axis_normalized = z_axis / (np.linalg.norm(z_axis))
            
            coordinate_systems[joint] = (x_axis_normalized, y_axis_normalized, z_axis_normalized)
    
    # 3. Compute rotation matrices and convert to axis-angle
    axis_angle_params = []
    
    for finger in fingers:
        for i, joint in enumerate(finger):
            if i == 0:
                # Root joint: parent is wrist
                parent_coords = C_wrist
            else:
                parent_coords = coordinate_systems[finger[i-1]]
            
            current_coords = coordinate_systems[joint]
            
            # Calculate rotation matrix components
            R = np.zeros((3, 3))
            for col in range(3):
                current_axis = current_coords[col]
                R[0, col] = np.dot(current_axis, parent_coords[0])
                R[1, col] = np.dot(current_axis, parent_coords[1])
                R[2, col] = np.dot(current_axis, parent_coords[2])
            
            # Convert to axis-angle
            rotation_vector, _ = cv2.Rodrigues(R)
            axis_angle_params.extend(rotation_vector.flatten())
    
    return axis_angle_params[:45]  # Ensure 45 parameters

# Example usage
# Assuming 'points' is a numpy array of shape (21, 3)
# axis_angle = compute_axis_angle_params(points)