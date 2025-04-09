#!/usr/bin/env python3

import sys
sys.path.append("../..")
from HandTrackerRenderer import HandTrackerRenderer
from Filters import LandmarksSmoothingFilter
import argparse
import numpy as np
import cv2
from o3d_utils import Visu3D
from joint_angles_sender import JointAnglesSender

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

class HandTracker3DRenderer:
    def __init__(self, tracker, mode_3d="image", smoothing=True):

        self.tracker = tracker
        self.mode_3d = mode_3d
        self.time = 0
        if self.mode_3d == "mixed" and not self.tracker.xyz:
            print("'mixed' 3d visualization needs the tracker to be in 'xyz' mode !")
            print("3d visualization falling back to 'world' mode.")
            self.mode_3d = 'world'
        if self.mode_3d == "image":
            self.vis3d = Visu3D(zoom=0.7, segment_radius=10)
            z = min(tracker.img_h, tracker.img_w)/3
            self.vis3d.create_grid([0,tracker.img_h,-z],[tracker.img_w,tracker.img_h,-z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Floor
            self.vis3d.create_grid([0,0,z],[tracker.img_w,0,z],[tracker.img_w,tracker.img_h,z],[0,tracker.img_h,z],5,2) # Wall
            self.vis3d.init_view()
        elif "world" in self.mode_3d:
            self.vis3d = Visu3D(bg_color=(0.2, 0.2, 0.2), zoom=1.1, segment_radius=0.01)
            x_max = 0.2 if self.tracker.solo else 0.4
            y_max = 0.2
            z_max = 0.2
            self.vis3d.create_grid([-x_max,y_max,-z_max],[x_max,y_max,-z_max],[x_max,y_max,z_max],[-x_max,y_max,z_max],1 if self.tracker.solo else 2,1) # Floor
            self.vis3d.create_grid([-x_max,y_max,z_max],[x_max,y_max,z_max],[x_max,-y_max,z_max],[-x_max,-y_max,z_max],1 if self.tracker.solo else 2,1) # Wall
            self.vis3d.init_view()
        elif self.mode_3d == "mixed":
            self.vis3d = Visu3D(bg_color=(0.4, 0.4, 0.4), zoom=0.8, segment_radius=0.01)
            x_max = 0.9
            y_max = 0.6
            grid_depth = 2
            self.vis3d.create_grid([-x_max,y_max,0],[x_max,y_max,0],[x_max,y_max,grid_depth],[-x_max,y_max,grid_depth],2,grid_depth) # Floor
            self.vis3d.create_grid([-x_max,y_max,grid_depth],[x_max,y_max,grid_depth],[x_max,-y_max,grid_depth],[-x_max,-y_max,grid_depth],2,2) # Wall
            self.vis3d.create_camera()
            self.vis3d.init_view()

        self.smoothing = smoothing
        self.filter = None
        if self.smoothing:
            if tracker.solo:
                if self.mode_3d == "image":
                    self.filter = [LandmarksSmoothingFilter(min_cutoff=0.01, beta=40, derivate_cutoff=1)]
                else:
                    self.filter = [LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True)]
            else:
                if self.mode_3d == "image":
                    self.filter = [
                        LandmarksSmoothingFilter(min_cutoff=0.01,beta=40,derivate_cutoff=1),
                        LandmarksSmoothingFilter(min_cutoff=0.01,beta=40,derivate_cutoff=1)
                        ]
                else:
                    self.filter = [
                        LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True),
                        LandmarksSmoothingFilter(min_cutoff=1, beta=20, derivate_cutoff=10, disable_value_scaling=True)
                        ]

        self.nb_hands_in_previous_frame = 0

    def calculate_joint_angles(self, landmarks):
        """
        计算手部各关节角度
        landmarks: 手部关节的3D坐标 (世界坐标系)
        返回: 包含各关节角度的字典
        """
        # 定义手指关节索引 (MediaPipe手部模型)
        # 手腕: 0
        # 拇指: 1-4 (1:掌指关节, 2:近节指关节, 3:远节指关节, 4:指尖)
        # 食指: 5-8
        # 中指: 9-12
        # 无名指: 13-16
        # 小指: 17-20
        
        # 计算向量
        def calculate_vector(point1, point2):
            return point2 - point1
        
        # 计算两个向量之间的角度(弧度)
        def calculate_angle(v1, v2):
            dot_product = np.dot(v1, v2)
            norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
            # 避免除零错误
            if norm_product < 1e-6:
                return 0
            # 确保结果在-1到1之间，避免数值误差
            cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
            angle_rad = np.arccos(cos_angle)
            return angle_rad * 180 / np.pi  # 转换为角度
        
        angles = {}
        
        # 计算拇指角度
        # 掌指关节(CMC)
        v1 = calculate_vector(landmarks[0], landmarks[1])
        v2 = calculate_vector(landmarks[1], landmarks[2]) # cmc is not that obvious, so use tip instead
        angles['thumb_cmc'] = calculate_angle(v1, v2)
        
        # 近节指关节(MCP)
        v1 = calculate_vector(landmarks[1], landmarks[2])
        v2 = calculate_vector(landmarks[2], landmarks[3])
        angles['thumb_mcp'] = (calculate_angle(v1, v2) - 0) * 1 # no obvious, multiply 2
        
        # 远节指关节(IP)
        v1 = calculate_vector(landmarks[2], landmarks[3])
        v2 = calculate_vector(landmarks[3], landmarks[4])
        angles['thumb_ip'] = (calculate_angle(v1, v2) - 0) # ignore
        
        # 计算其他手指的角度
        finger_names = ['index', 'middle', 'ring', 'little']
        for i, name in enumerate(finger_names):
            base_idx = 5 + i * 4
            
            # 掌指关节(MCP)
            v1 = calculate_vector(landmarks[0], landmarks[base_idx])
            v2 = calculate_vector(landmarks[base_idx], landmarks[base_idx + 1])
            angles[f'{name}_mcp'] = (calculate_angle(v1, v2) - 0) * 1 # wrist and mcp not in verticle line
            
            # 近节指关节(PIP)
            v1 = calculate_vector(landmarks[base_idx], landmarks[base_idx + 1])
            v2 = calculate_vector(landmarks[base_idx + 1], landmarks[base_idx + 2])
            angles[f'{name}_pip'] = calculate_angle(v1, v2) * 1
            
            # 远节指关节(DIP)
            v1 = calculate_vector(landmarks[base_idx + 1], landmarks[base_idx + 2])
            v2 = calculate_vector(landmarks[base_idx + 2], landmarks[base_idx + 3])
            angles[f'{name}_dip'] = calculate_angle(v1, v2) # ignore

        # 添加手腕姿态角度计算
        # 定义手掌平面和方向
        wrist = landmarks[0]  # 手腕点
        index_mcp = landmarks[5]  # 食指掌指关节
        pinky_mcp = landmarks[17]  # 小指掌指关节
        middle_mcp = landmarks[9]  # 中指掌指关节
        
        # 计算手掌平面的法向量
        palm_vector1 = index_mcp - wrist
        palm_vector2 = pinky_mcp - wrist
        palm_normal = np.cross(palm_vector1, palm_vector2)
        palm_normal = palm_normal / np.linalg.norm(palm_normal)
        
        # 计算手指方向向量 (从手腕指向中指MCP)
        finger_direction = middle_mcp - wrist
        finger_direction = finger_direction / np.linalg.norm(finger_direction)
        
        # 计算手掌侧向向量 (垂直于手指方向和手掌法向量)
        palm_side = np.cross(finger_direction, palm_normal)
        palm_side = palm_side / np.linalg.norm(palm_side)
        
        # OAK世界坐标系为：
        # x轴：向右
        # y轴：向上
        # z轴：向前
        world_up = np.array([0, 1, 0])
        world_forward = np.array([0, 0, 1])
        
        # 计算俯仰角 (pitch) - 手掌抬起/低下的角度
        # 俯仰角是手指方向向量在y-z平面上的投影与y轴的夹角
        finger_yz = np.array([0, finger_direction[1], finger_direction[2]])
        if np.linalg.norm(finger_yz) > 1e-6:
            finger_yz = finger_yz / np.linalg.norm(finger_yz)
            pitch = np.arccos(np.clip(np.dot(finger_yz, world_up), -1.0, 1.0))
            # 调整符号：当手指指向前方时，俯仰角为正
            if finger_direction[2] < 0:
                pitch = -pitch
        else:
            pitch = 0
        
        # 计算横滚角 (roll) - 手掌左右倾斜的角度
        # 横滚角是手指方向量在x-y平面上的投影与y轴的夹角
        finger_xy = np.array([finger_direction[0], finger_direction[1], 0])
        if np.linalg.norm(finger_xy) > 1e-6:
            finger_xy = finger_xy / np.linalg.norm(finger_xy)
            roll = np.arccos(np.clip(np.dot(finger_xy, world_up), -1.0, 1.0))
            # 调整符号：当手指指向前方时，俯仰角为正
            if finger_direction[0] < 0:
                roll = -roll
        else:
            roll = 0
        '''
        # 计算横滚角 (roll) - 手掌左右倾斜的角度
        # 横滚角是手掌法向量在x-y平面上的投影与y轴的夹角
        palm_normal_xy = np.array([palm_normal[0], palm_normal[1], 0])
        if np.linalg.norm(palm_normal_xy) > 1e-6:
            palm_normal_xy = palm_normal_xy / np.linalg.norm(palm_normal_xy)
            roll = np.arccos(np.clip(np.dot(palm_normal_xy, world_up), -1.0, 1.0))
            # 调整符号：当手掌法向量指向右侧时，横滚角为正
            if palm_normal[0] > 0:
                roll = -roll
        else:
            roll = 0
        
        # 计算偏航角 (yaw) - 手掌左右转动的角度
        # 偏航角是手指方向向量在x-z平面上的投影与z轴的夹角
        finger_xz = np.array([finger_direction[0], 0, finger_direction[2]])
        if np.linalg.norm(finger_xz) > 1e-6:
            finger_xz = finger_xz / np.linalg.norm(finger_xz)
            yaw = np.arccos(np.clip(np.dot(finger_xz, world_forward), -1.0, 1.0))
            # 调整符号：当手指指向右侧时，偏航角为正
            if finger_direction[0] > 0:
                yaw = -yaw
        else:
            yaw = 0
        '''
        
        # 转换为角度
        angles['wrist_pitch'] = pitch * 180 / np.pi
        angles['wrist_roll'] = roll * 180 / np.pi

        return angles
    
    def draw_hand(self, hand, i):
        if self.mode_3d == "image":
            # Denormalize z-component of 'norm_landmarks'
            lm_z = (hand.norm_landmarks[:,2:3] * hand.rect_w_a  / 0.4).astype(np.int32)
            # ... and concatenates with x and y components of 'landmarks'
            points = np.hstack((hand.landmarks, lm_z))
            radius = hand.rect_w_a / 30 # Thickness of segments depends on the hand size
        elif "world" in self.mode_3d:
            if self.mode_3d == "raw_world":
                points = hand.world_landmarks
            else: # "world"
                points = hand.get_rotated_world_landmarks()
            if not self.tracker.solo:
                delta_x = -0.2 if  hand.label == "right" else 0.2
                points = points + np.array([delta_x,0,0])
            radius = 0.01
        elif self.mode_3d == "mixed":
            wrist_xyz = hand.xyz / 1000.0
            # Beware that y value of (x,y,z) coordinates given by depth sensor is negative 
            # in the lower part of the image and positive in the upper part.
            wrist_xyz[1] = -wrist_xyz[1]
            points = hand.get_rotated_world_landmarks()
            points = points + wrist_xyz - points[0]
            radius = 0.01

        if self.smoothing:
            points = self.filter[i].apply(points, object_scale=hand.rect_w_a)
            hand.joint_angles = self.calculate_joint_angles(points)
            if self.time % 20 == 0:
                if hasattr(hand, 'joint_angles'):
                    print(f"========================================\n")
                    for key in hand.joint_angles:
                        print(f"==> {key}: {hand.joint_angles[key]}")
                    print(f"\n========================================\n")
            self.time += 1

        for i,a_b in enumerate(LINES_HAND):
            a, b = a_b
            self.vis3d.add_segment(points[a], points[b], radius=radius, color=[1*(1-hand.handedness),hand.handedness,0]) # if hand.handedness<0.5 else [0,1,0])
                    
    def draw(self, hands):
        if self.smoothing and len(hands) != self.nb_hands_in_previous_frame:
            for f in self.filter: f.reset()
        self.vis3d.clear()
        self.vis3d.try_move()
        self.vis3d.add_geometries()
        for i, hand in enumerate(hands):
            self.draw_hand(hand, i)
        self.vis3d.render()
        self.nb_hands_in_previous_frame = len(hands)

parser = argparse.ArgumentParser()
# parser.add_argument('-e', '--edge', action="store_true",
#                     help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument('-i', '--input', type=str, 
                    help="Path to video or image file to use as input (if not specified, use OAK color camera)")
parser_tracker.add_argument("--pd_model", type=str,
                    help="Path to a blob file for palm detection model")
parser_tracker.add_argument("--lm_model", type=str,
                    help="Path to a blob file for landmark model")
parser_tracker.add_argument('-s', '--solo', action="store_true", 
                    help="Detect one hand max")         
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int,                                                                                 
                    help="Internal color camera frame height in pixels")    
parser_tracker.add_argument('--single_hand_tolerance_thresh', type=int, default=0,
                    help="(Duo mode only) Number of frames after only one hand is detected before calling palm detection (default=%(default)s)")                

parser_renderer3d = parser.add_argument_group("3D Renderer arguments")
parser_renderer3d.add_argument('-m', '--mode_3d', nargs='?', 
                    choices=['image', 'world', 'raw_world', 'mixed'], const='image', default='image',
                    help="Specify the 3D coordinates used. See README for description (default=%(default)s)")
parser_renderer3d.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter (smoothing works only in solo mode)")   
parser.add_argument('--server_ip', type=str, default='172.30.83.97',
                    help="WSL2 服务器IP地址")
parser.add_argument('--server_port', type=int, default=12346,
                    help="WSL2 服务器端口")
args = parser.parse_args()

args.edge = True
if args.edge:
    from HandTrackerEdge import HandTracker
else:
    from HandTracker import HandTracker

dargs = vars(args)
tracker_args = {a:dargs[a] for a in ['pd_model', 'lm_model', 'internal_fps', 'internal_frame_height'] if dargs[a] is not None}

tracker = HandTracker(
        input_src=args.input, 
        use_world_landmarks=args.mode_3d != "image",
        solo=args.solo,
        xyz= args.mode_3d == "mixed",
        stats=True,
        single_hand_tolerance_thresh=args.single_hand_tolerance_thresh,
        lm_nb_threads=1,
        **tracker_args
        )

renderer3d = HandTracker3DRenderer(tracker, mode_3d=args.mode_3d, smoothing=not args.no_smoothing)
renderer2d = HandTrackerRenderer(tracker)

# 初始化关节角度发送器
joint_angles_sender = JointAnglesSender(server_ip=args.server_ip, server_port=args.server_port)

pause = False
hands = []

while True:
    # Run hand tracker on next frame
    if not pause:
        frame, hands, bag = tracker.next_frame()
        if frame is None: break
        # Render 2d frame
        frame = renderer2d.draw(frame, hands, bag)
        cv2.imshow("HandTracker", frame)
    key = cv2.waitKey(1)
    # Draw hands on open3d canvas
    renderer3d.draw(hands)
    
    # 发送关节角度数据
    for hand in hands:
        if hasattr(hand, 'joint_angles'):
            hand_type = "left" if hand.handedness < 0.5 else "right"
            joint_angles_sender.send_joint_angles(hand_type, hand.joint_angles)
    
    if key == 27 or key == ord('q'):
        break
    elif key == 32: # space
        pause = not pause
    elif key == ord('s'):
        if renderer3d.filter:
            renderer3d.smoothing = not renderer3d.smoothing

# 关闭发送器
joint_angles_sender.close()
renderer2d.exit()
tracker.exit()
