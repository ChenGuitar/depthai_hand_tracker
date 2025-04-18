#!/usr/bin/env python3

import sys
sys.path.append("../..")
from HandTrackerRenderer import HandTrackerRenderer
from Filters import LandmarksSmoothingFilter
import argparse
import numpy as np
import cv2
from o3d_utils import Visu3D
import torch
import smplx
from joint_angles_sender import JointAnglesSender

# 添加MANO模型路径
MANO_MODEL_PATH = 'D:/Users/11235/Desktop/OAKChina/depthai_ws/depthai_hand_tracker/models/'  # 替换为实际路径

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
        # 加载MANO模型（左右手）
        self.mano_models = {
            'left': smplx.create(MANO_MODEL_PATH, model_type='mano', use_pca=False, is_rhand=False),
            'right': smplx.create(MANO_MODEL_PATH, model_type='mano', use_pca=False, is_rhand=True)
        }

        self.mediapipe_indices = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
        self.mano_joint_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]    

        self.nb_hands_in_previous_frame = 0

    def process_hand_with_mano(self, hand_type, world_landmarks):
        # 获取旋转后的世界坐标系关键点
        wrist_pos = world_landmarks[0]
        target_positions = world_landmarks[self.mediapipe_indices]

        # 确定左右手
        mano_model = self.mano_models[hand_type]

        # 转换到Tensor
        transl = torch.tensor(wrist_pos, dtype=torch.float32).unsqueeze(0)
        target_positions_tensor = torch.tensor(target_positions, dtype=torch.float32).unsqueeze(0)

        # 初始化姿态参数（排除手腕，优化后45维）
        pose_params = torch.zeros(1, 45, requires_grad=True)
        optimizer = torch.optim.Adam([pose_params], lr=0.1)

        # 优化循环
        for _ in range(100):
            optimizer.zero_grad()
            # 组合姿态参数（手腕为0，后面是优化参数）
            full_pose = torch.cat([torch.zeros(1, 3), pose_params], dim=1)
            output = mano_model(pose=full_pose, transl=transl, betas=torch.zeros(1, 10))
            joints = output.joints[:, self.mano_joint_indices, :]
            loss = torch.mean((joints - target_positions_tensor) ** 2)
            loss.backward()
            optimizer.step()

        # 提取轴角参数（15*3）
        axis_angles = pose_params.detach().numpy().reshape(15, 3)
        return axis_angles
    
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
            hand_type = "left" if hand.handedness < 0.5 else "right"
            hand.joint_angles = self.process_hand_with_mano(hand_type, points)
            hand.joint_angles = -np.degrees(hand.joint_angles)
             # 发送关节角度数据
            hand_type = "left" if hand.handedness < 0.5 else "right"
            joint_angles_sender.send_joint_angles(hand_type, hand.joint_angles) # 发送关节角度数据到目标服务器

            if self.time % 20 == 0:
                if hasattr(hand, 'joint_angles'):
                    print(f"\n===== {hand.label.upper()} HAND JOINTS =====")
                    print(f"Hand {i} Axis-Angle Parameters:\n{hand.joint_angles}")
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
parser.add_argument('--server_ip', type=str, default='127.0.0.1',
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
