import os

os.environ["SDL_VIDEO_WINDOW_POS"] = "30,60"

import copy

import numpy as np

import pygame

import mujoco

import mujoco.viewer

# ============================================================

# CONFIG

# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XML_PATH = os.path.join(BASE_DIR, "main_vertical.xml")

WINDOW_W = 1400

ROBOT_VIEW_W = 820
STATUS_PANEL_W = WINDOW_W - ROBOT_VIEW_W
status_panel_next_y = 0

WINDOW_H = 900

FPS = 60

# ============================================================

# LIVE MUJOCO PREVIEW

# ============================================================

LIVE_PREVIEW = True

# ============================================================

# EDITOR CAMERA / ZOOM

# ============================================================

SCALE = 1500.0

MIN_SCALE = 100.0

MAX_SCALE = 5000.0

ZOOM_FACTOR = 1.15

SCREEN_CENTER_X = ROBOT_VIEW_W // 2

SCREEN_CENTER_Y = WINDOW_H // 2

IK_ITERATIONS = 25

IK_EPS = 1e-4

IK_MAX_STEP = 0.06

IK_TOLERANCE_PX = 2.0

WHOLE_BODY_ITERATIONS = 35

WHOLE_BODY_EPS = 1e-4

WHOLE_BODY_MAX_STEP = 0.025

POINT_RADIUS = 10

BODY_CENTER_RADIUS = 17

HIT_RADIUS = 22

# ============================================================

# POSE ANIMATION

# ============================================================

PLAYING = False

# Pose A -> Pose B 所需時間（秒）

POSE_DURATION = 1.5

# 播放狀態

play_segment = 0

play_t = 0.0

# ============================================================

# PHYSICS PLAYBACK

# ============================================================

# F toggles KINEMATIC / PHYSICS playback.

PHYSICS_MODE = False

PHYSICS_PLAYING = False

# How long each saved target is commanded before moving on.

PHYSICS_POSE_DURATION = 1.5

physics_segment = 0

physics_segment_t = 0.0

# In PHYSICS MODE the mouse still edits working_pose.

# The pose is sent to position actuators and MuJoCo advances with mj_step().

# This lets you design Pose 1, Pose 2... while gravity/contact are active.

PHYSICS_EDITING = False

# Gravity is read from test.xml. Manual wrist control has no automatic attitude feedback.

# ============================================================

# COLORS

# ============================================================

BG = (25, 27, 32)

GRID = (45, 48, 55)

WHITE = (245, 245, 245)

GREY = (160, 160, 160)

GREEN = (60, 230, 100)

BLUE = (70, 150, 255)

RED = (255, 40, 40)

YELLOW = (255, 210, 50)

MAGENTA = (230, 80, 230)

LINK_COLOR = (70, 230, 110)

# ============================================================

# LOAD MUJOCO

# ============================================================

if not os.path.exists(XML_PATH):

    raise FileNotFoundError(f"Cannot find:\n{XML_PATH}")

model = mujoco.MjModel.from_xml_path(XML_PATH)

sim_data = mujoco.MjData(model)

edit_data = mujoco.MjData(model)

mujoco.mj_resetData(model, sim_data)

mujoco.mj_resetData(model, edit_data)

mujoco.mj_forward(model, sim_data)

mujoco.mj_forward(model, edit_data)

# ============================================================

# SAVE RL HANG START STATE

# 儲存「按 F 的瞬間、重力尚未讓 robot 掉落」的完整狀態

# ============================================================

hang_qpos_path = os.path.join(BASE_DIR, "hang_start_qpos.npy")

hang_ctrl_path = os.path.join(BASE_DIR, "hang_start_ctrl.npy")

np.save(hang_qpos_path, sim_data.qpos.copy())

np.save(hang_ctrl_path, sim_data.ctrl.copy())

print("\n========================================")

print("RL HANG START STATE SAVED")

print("qpos:", hang_qpos_path)

print("ctrl:", hang_ctrl_path)

print("nq =", model.nq)

print("nu =", model.nu)

print("========================================\n")

# ============================================================

# JOINTS

# ============================================================

JOINTS = {

    "R_shoulder_pitch": "torso_R_shoulder_pitch",

    "R_shoulder_roll": "R_shoulder_R_shoulder_roll",

    "R_elbow_roll": "R_upper_arm_R_elbow_roll",

    "R_wrist_pitch": "R_forearm_R_wrist_pitch",

    "L_shoulder_pitch": "torso_L_shoulder_pitch",

    "L_shoulder_roll": "L_shoulder_L_shouldr_roll",

    "L_elbow_roll": "L_upper_arm_L_elbow_roll",

    "L_wrist_pitch": "L_forearm_L_wrist_pitch",

    "R_pelvis_roll": "pelvis_R_pelvis_roll",

    "R_hip_yaw": "R_upper_hip_R_upper_hip_yaw",

    "R_hip_roll": "R_lower_hip_R_lower_hip_roll",

    "R_knee_roll": "R_thigh_R_knee_roll",

    "L_pelvis_roll": "pelvis_L_pelvis_roll",

    "L_hip_yaw": "L_upper_hip_L_upper_hip_yaw",

    "L_hip_roll": "L_lower_hip_L_lower_hip_roll",

    "L_knee_roll": "L_thigh_L_knee_roll",

}

# ============================================================

# MUJOCO IDs

# ============================================================

def body_id(name):

    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    if i < 0:

        raise RuntimeError(f"Body not found: {name}")

    return i

def site_id(name):

    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

    if i < 0:

        raise RuntimeError(f"Site not found: {name}")

    return i

joint_ids = {}

joint_qpos_adr = {}

for key, name in JOINTS.items():

    jid = mujoco.mj_name2id(

        model,

        mujoco.mjtObj.mjOBJ_JOINT,

        name

    )

    if jid < 0:

        raise RuntimeError(f"Joint not found: {name}")

    joint_ids[key] = jid

    joint_qpos_adr[key] = int(model.jnt_qposadr[jid])

# ============================================================

# ROBOT BODY / SITE

# ============================================================

BODY = {

    "pelvis": body_id("pelvis"),

    "torso": body_id("torso"),

    # Arms

    "L_shoulder": body_id("L_upper_arm"),

    "R_shoulder": body_id("R_upper_arm"),

    "L_elbow": body_id("L_forearm"),

    "R_elbow": body_id("R_forearm"),

    # Knee body

    "L_knee": body_id("L_calf"),

    "R_knee": body_id("R_calf"),

}

SITE = {

    "L_hand": site_id("L_hand_site"),

    "R_hand": site_id("R_hand_site"),

    "L_foot": site_id("L_foot_site_front_L"),

    "R_foot": site_id("R_foot_site_front_L"),

}

END_EFFECTORS = [

    "L_hand",

    "R_hand",

    "L_foot",

    "R_foot",

]

# ============================================================

# FREE ROOT

# ============================================================

ROOT_JOINT_ID = mujoco.mj_name2id(

    model,

    mujoco.mjtObj.mjOBJ_JOINT,

    "robot_free_root"

)

ROOT_QPOS_ADR = int(

    model.jnt_qposadr[ROOT_JOINT_ID]

)

# freejoint qpos:

# [x, y, z, qw, qx, qy, qz]

ROOT_X = ROOT_QPOS_ADR

ROOT_Y = ROOT_QPOS_ADR + 1

ROOT_Z = ROOT_QPOS_ADR + 2

# ============================================================

# POSE

# 現在 Pose 同時保存 root X/Y/Z

# ============================================================

def create_zero_pose():

    pose = {

        "root_x": float(model.qpos0[ROOT_X]),

        "root_y": float(model.qpos0[ROOT_Y]),

        "root_z": float(model.qpos0[ROOT_Z]),

    }

    for key in JOINTS:

        pose[key] = float(

            model.qpos0[joint_qpos_adr[key]]

        )

    return pose

# ============================================================

# POSE RECORDER

#

# initial_pose : 程式啟動時的初始姿勢，ENTER 播放一定從這裡開始

# poses        : 已儲存的 Pose 1, Pose 2, Pose 3...

# working_pose : 目前正在編輯的姿勢

#

# N = 儲存目前 working_pose 為下一個 Pose

# 儲存後 working_pose 不會 reset，因此可以從上一個 Pose 繼續編輯

# ============================================================

initial_pose = create_zero_pose()

poses = []

MAX_POSES = 100

working_pose = copy.deepcopy(initial_pose)

current_pose_index = -1

# -1 代表目前正在編輯 working_pose；

# 0,1,2... 代表用 4/5 查看已儲存 Pose。

current_pose_index = -1

# ============================================================

# JOINT LIMIT

# ============================================================

def clamp_joint(key, value):

    jid = joint_ids[key]

    if model.jnt_limited[jid]:

        low = model.jnt_range[jid][0]

        high = model.jnt_range[jid][1]

        return float(np.clip(value, low, high))

    return float(value)

# ============================================================

# APPLY POSE

# ============================================================

def apply_pose_to_edit_data(pose):

    edit_data.qpos[:] = model.qpos0

    edit_data.qvel[:] = 0

    # Root translation

    edit_data.qpos[ROOT_X] = pose["root_x"]

    edit_data.qpos[ROOT_Y] = pose["root_y"]

    edit_data.qpos[ROOT_Z] = pose["root_z"]

    # Robot joints

    for key in JOINTS:

        edit_data.qpos[

            joint_qpos_adr[key]

        ] = pose[key]

    mujoco.mj_forward(model, edit_data)

apply_pose_to_edit_data(working_pose)

# ============================================================

# WORLD -> SCREEN

# ============================================================

initial_pelvis_world = edit_data.xpos[

    BODY["pelvis"]

].copy()

def get_skeleton_world_center():

    """Center the camera on the robot's visible joint/site bounds."""

    positions = np.vstack([

        edit_data.xpos[list(BODY.values())],

        edit_data.xanchor[list(joint_ids.values())],

        edit_data.site_xpos[list(SITE.values())],

    ])

    return (positions.min(axis=0) + positions.max(axis=0)) / 2.0

camera_world_center = get_skeleton_world_center()

def fit_robot_view():
    global SCALE, camera_world_center
    data = sim_data if PHYSICS_MODE else edit_data
    attitude_data.qpos[:] = data.qpos
    mujoco.mj_kinematics(model, attitude_data)
    positions = np.vstack([
        attitude_data.xpos[list(BODY.values())],
        attitude_data.xanchor[list(joint_ids.values())],
        attitude_data.site_xpos[list(SITE.values())],
    ])
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    camera_world_center = (lo + hi) / 2.0
    width, height = max(float(hi[0] - lo[0]), 0.01), max(float(hi[2] - lo[2]), 0.01)
    SCALE = float(np.clip(min((ROBOT_VIEW_W - 120) / width,
                              (WINDOW_H - 140) / height), MIN_SCALE, MAX_SCALE))


def world_to_screen(world_pos):

    x = world_pos[0]

    z = world_pos[2]

    sx = (

        SCREEN_CENTER_X

        + (x - camera_world_center[0]) * SCALE

    )

    sy = (

        SCREEN_CENTER_Y

        - (z - camera_world_center[2]) * SCALE

    )

    return np.array(

        [sx, sy],

        dtype=float

    )

# ============================================================

# SCREEN DELTA -> WORLD DELTA

# Body Center 只控制 X/Z

# ============================================================

def screen_delta_to_world(dx, dy):

    world_dx = dx / SCALE

    world_dz = -dy / SCALE

    return world_dx, world_dz

# ============================================================

# JOINT WORLD POSITION

# ============================================================

def joint_world_pos(key):

    jid = joint_ids[key]

    return edit_data.xanchor[jid].copy()

# ============================================================

# SKELETON POINTS

# A = pelvis_roll

# B = hip_roll

# C = knee_roll

# ============================================================

def get_skeleton_points():

    points = {}

    # Arms

    points["L_shoulder"] = world_to_screen(

        edit_data.xpos[BODY["L_shoulder"]]

    )

    points["R_shoulder"] = world_to_screen(

        edit_data.xpos[BODY["R_shoulder"]]

    )

    points["L_elbow"] = world_to_screen(

        edit_data.xpos[BODY["L_elbow"]]

    )

    points["R_elbow"] = world_to_screen(

        edit_data.xpos[BODY["R_elbow"]]

    )

    # A: pelvis_roll 軸心

    points["L_pelvis_roll"] = world_to_screen(

        joint_world_pos("L_pelvis_roll")

    )

    points["R_pelvis_roll"] = world_to_screen(

        joint_world_pos("R_pelvis_roll")

    )

    # B: hip_roll 軸心

    points["L_hip"] = world_to_screen(

        joint_world_pos("L_hip_roll")

    )

    points["R_hip"] = world_to_screen(

        joint_world_pos("R_hip_roll")

    )

    # C: knee_roll 軸心

    points["L_knee"] = world_to_screen(

        joint_world_pos("L_knee_roll")

    )

    points["R_knee"] = world_to_screen(

        joint_world_pos("R_knee_roll")

    )

    # End effectors

    for name, sid in SITE.items():

        points[name] = world_to_screen(edit_data.site_xpos[sid])

    # Body Center = 左右 Shoulder 中點

    points["body_center"] = (

        points["L_shoulder"] + points["R_shoulder"]

    ) / 2

    return points

# ============================================================

# GET END EFFECTOR WORLD POSITIONS

# ============================================================

def get_end_effector_targets():

    targets = {}

    for name in END_EFFECTORS:

        targets[name] = (

            edit_data.site_xpos[

                SITE[name]

            ].copy()

        )

    return targets

# ============================================================

# NORMAL END-EFFECTOR IK

# 用於直接拖 Hand / Foot

# ============================================================

IK_CHAINS = {

    "L_hand": [

        "L_shoulder_roll",

        "L_elbow_roll",

    ],

    "R_hand": [

        "R_shoulder_roll",

        "R_elbow_roll",

    ],

    "L_foot": [

        "L_hip_roll",

        "L_knee_roll",

    ],

    "R_foot": [

        "R_hip_roll",

        "R_knee_roll",

    ],

}

def get_endpoint_screen(name):

    return world_to_screen(

        edit_data.site_xpos[SITE[name]]

    )

def solve_ik(endpoint, target_screen, pose):

    chain = IK_CHAINS[endpoint]

    target_screen = np.array(target_screen, dtype=float)

    for _ in range(IK_ITERATIONS):

        apply_pose_to_edit_data(pose)

        current = get_endpoint_screen(endpoint)

        error = target_screen - current

        if np.linalg.norm(error) < IK_TOLERANCE_PX:

            break

        J = np.zeros((2, len(chain)))

        for col, key in enumerate(chain):

            original = pose[key]

            test = clamp_joint(

                key,

                original + IK_EPS

            )

            de = test - original

            if abs(de) < 1e-8:

                test = clamp_joint(

                    key,

                    original - IK_EPS

                )

                de = test - original

            if abs(de) < 1e-8:

                continue

            pose[key] = test

            apply_pose_to_edit_data(pose)

            moved = get_endpoint_screen(endpoint)

            J[:, col] = (

                moved - current

            ) / de

            pose[key] = original

        apply_pose_to_edit_data(pose)

        dq = np.linalg.pinv(J) @ error

        dq = np.clip(

            dq,

            -IK_MAX_STEP,

            IK_MAX_STEP

        )

        for i, key in enumerate(chain):

            pose[key] = clamp_joint(

                key,

                pose[key] + dq[i]

            )

    apply_pose_to_edit_data(pose)

# ============================================================

# HIP DRAG IK

# A = pelvis_roll

# B = hip

# ============================================================

HIP_JOINT = {

    "L_hip":"L_pelvis_roll",

    "R_hip":"R_pelvis_roll",

}

def get_hip_screen(name):

    key = "L_hip_roll" if name=="L_hip" else "R_hip_roll"

    return world_to_screen(joint_world_pos(key))

def solve_hip_drag(name,target,pose):

    joint = HIP_JOINT[name]

    target = np.array(target,float)

    for _ in range(IK_ITERATIONS):

        apply_pose_to_edit_data(pose)

        cur = get_hip_screen(name)

        err = target-cur

        if np.linalg.norm(err)<IK_TOLERANCE_PX:

            break

        q0 = pose[joint]

        # 先測試正方向；若碰到 joint limit，再自動測試負方向。

        # 這對左右 pelvis 的鏡像 joint range 很重要：

        # L pelvis 可能允許正角度，R pelvis 可能允許負角度。

        q1 = clamp_joint(joint, q0 + IK_EPS)

        dq = q1 - q0

        if abs(dq) < 1e-8:

            q1 = clamp_joint(joint, q0 - IK_EPS)

            dq = q1 - q0

        if abs(dq) < 1e-8:

            break

        pose[joint]=q1

        apply_pose_to_edit_data(pose)

        nxt = get_hip_screen(name)

        J = (nxt-cur)/dq

        pose[joint]=q0

        apply_pose_to_edit_data(pose)

        step = np.dot(J,err)/(np.dot(J,J)+1e-8)

        step = np.clip(step,-IK_MAX_STEP,IK_MAX_STEP)

        pose[joint]=clamp_joint(joint,q0+step)

    apply_pose_to_edit_data(pose)

# ============================================================

# WHOLE-BODY IK JOINTS

# Arms: shoulder roll + elbow

# Legs: pelvis roll + hip roll + knee

# ============================================================

WHOLE_BODY_JOINTS = [

    "L_shoulder_roll",

    "L_elbow_roll",

    "R_shoulder_roll",

    "R_elbow_roll",

    # Left leg

    "L_pelvis_roll",

    "L_hip_roll",

    "L_knee_roll",

    # Right leg

    "R_pelvis_roll",

    "R_hip_roll",

    "R_knee_roll",

]

# ============================================================

# WHOLE-BODY ERROR

#

# 四個 End Effectors：

# 每個只鎖 X/Z

#

# error =

# L hand X,Z

# R hand X,Z

# L foot X,Z

# R foot X,Z

#

# 共 8 個 constraint

# ============================================================

def whole_body_error(targets):

    error = []

    for name in END_EFFECTORS:

        current = edit_data.site_xpos[

            SITE[name]

        ]

        target = targets[name]

        # X error

        error.append(

            target[0] - current[0]

        )

        # Z error

        error.append(

            target[2] - current[2]

        )

    return np.array(

        error,

        dtype=float

    )

# ============================================================

# WHOLE-BODY IK

#

# Root 已經移動到新的 X/Z。

# IK 調整 8 個 joints，

# 讓四個 end-effectors 回到原來 X/Z。

# ============================================================

def solve_whole_body(pose, targets):

    for _ in range(WHOLE_BODY_ITERATIONS):

        apply_pose_to_edit_data(pose)

        error = whole_body_error(targets)

        # 平均誤差小於約 0.5 mm

        if np.linalg.norm(error) < 0.0005:

            break

        J = np.zeros(

            (

                len(error),

                len(WHOLE_BODY_JOINTS)

            )

        )

        # Numerical Jacobian

        for col, key in enumerate(

            WHOLE_BODY_JOINTS

        ):

            original = pose[key]

            test = clamp_joint(

                key,

                original + WHOLE_BODY_EPS

            )

            de = test - original

            if abs(de) < 1e-8:

                test = clamp_joint(

                    key,

                    original - WHOLE_BODY_EPS

                )

                de = test - original

            if abs(de) < 1e-8:

                continue

            pose[key] = test

            apply_pose_to_edit_data(pose)

            moved_error = whole_body_error(

                targets

            )

            # error = target-current

            # derivative of current position

            # = -(new_error-old_error)/dq

            J[:, col] = (

                error - moved_error

            ) / de

            pose[key] = original

        apply_pose_to_edit_data(pose)

        # Damped Least Squares

        damping = 0.001

        A = (

            J.T @ J

            + damping

            * np.eye(len(WHOLE_BODY_JOINTS))

        )

        b = J.T @ error

        try:

            dq = np.linalg.solve(A, b)

        except np.linalg.LinAlgError:

            dq = np.linalg.pinv(J) @ error

        dq = np.clip(

            dq,

            -WHOLE_BODY_MAX_STEP,

            WHOLE_BODY_MAX_STEP

        )

        for i, key in enumerate(

            WHOLE_BODY_JOINTS

        ):

            pose[key] = clamp_joint(

                key,

                pose[key] + dq[i]

            )

    apply_pose_to_edit_data(pose)

# ============================================================

# ACTUATOR MAP FOR PHYSICS PLAYBACK

# ============================================================

# Find the position actuator that drives each editor joint.

JOINT_TO_ACTUATOR = {}

for key, jid in joint_ids.items():

    matches = np.where(

        model.actuator_trnid[:, 0] == jid

    )[0]

    if len(matches) > 0:

        JOINT_TO_ACTUATOR[key] = int(matches[0])

print("\nPHYSICS ACTUATOR MAP")

for key in JOINTS:

    if key in JOINT_TO_ACTUATOR:

        aid = JOINT_TO_ACTUATOR[key]

        aname = mujoco.mj_id2name(

            model,

            mujoco.mjtObj.mjOBJ_ACTUATOR,

            aid

        )

        print(f"  {key:20s} -> {aname}")

    else:

        print(f"  WARNING: no actuator for {key}")

# Auto Balance settings: angles in degrees, rate in degrees/second.
AUTO_BALANCE = False
SUPPORT_MODE = "BOTH"
BALANCE_TARGET_BACK_DEG = -2.0
BALANCE_DEADBAND_DEG = 1.5
BALANCE_KP = 0.4
BALANCE_MAX_CORRECTION_DEG = 8.0
BALANCE_RATE_DEG_S = 12.0
# Change each sign independently if the model pulls away from the wall.
BALANCE_SIGN = {"L_shoulder_pitch": 1.0, "R_shoulder_pitch": 1.0}
balance_correction_deg = 0.0
balance_owner = None
balance_manual = {}
balance_command = {}


def reset_balance():
    global balance_correction_deg, balance_owner
    balance_correction_deg = 0.0
    balance_owner = None


def update_balance(dt):
    global balance_correction_deg, balance_owner
    owner = {"LEFT_SUPPORT": "L_shoulder_pitch",
             "RIGHT_SUPPORT": "R_shoulder_pitch"}.get(SUPPORT_MODE)
    if not AUTO_BALANCE or not PHYSICS_MODE or owner is None:
        reset_balance()
        return
    if owner != balance_owner:
        reset_balance()
        balance_owner = owner
    attitude_data.qpos[:] = sim_data.qpos
    mujoco.mj_kinematics(model, attitude_data)
    back, _, _ = relative_attitude(torso_rotation(attitude_data), attitude_reference)
    if not np.isfinite(back):
        reset_balance()
        return
    error = back - BALANCE_TARGET_BACK_DEG
    # Continuous deadband avoids a jump at its boundary.
    error = np.sign(error) * max(0.0, abs(error) - BALANCE_DEADBAND_DEG)
    desired = float(np.clip(BALANCE_KP * error,
                            -BALANCE_MAX_CORRECTION_DEG,
                            BALANCE_MAX_CORRECTION_DEG))
    step = BALANCE_RATE_DEG_S * max(0.0, float(dt))
    balance_correction_deg += float(np.clip(desired - balance_correction_deg, -step, step))


def set_ctrl_from_pose(pose):
    """Add correction only to motor commands; never write into pose."""
    for key, aid in JOINT_TO_ACTUATOR.items():
        command = float(pose[key])
        if key in BALANCE_SIGN:
            balance_manual[key] = command
            if AUTO_BALANCE and PHYSICS_MODE and key == balance_owner:
                command += np.radians(balance_correction_deg * BALANCE_SIGN[key])
                command = clamp_joint(key, command)
                if model.actuator_ctrllimited[aid]:
                    command = float(np.clip(command, *model.actuator_ctrlrange[aid]))
            balance_command[key] = command
        sim_data.ctrl[aid] = command


def balance_manual_pose():
    pose = dict(working_pose)
    for key, aid in JOINT_TO_ACTUATOR.items():
        pose[key] = balance_manual.get(key, float(sim_data.ctrl[aid]))
    return pose


def reset_physics_to_initial():

    """

    Initialize physics ONCE at the editor initial pose.

    After this, qpos is not overwritten during physics playback.

    """

    reset_balance()

    mujoco.mj_resetData(model, sim_data)

    sim_data.qpos[:] = model.qpos0

    sim_data.qvel[:] = 0

    sim_data.qpos[ROOT_X] = initial_pose["root_x"]

    sim_data.qpos[ROOT_Y] = initial_pose["root_y"]

    sim_data.qpos[ROOT_Z] = initial_pose["root_z"]

    for key in JOINTS:

        sim_data.qpos[

            joint_qpos_adr[key]

        ] = initial_pose[key]

    set_ctrl_from_pose(initial_pose)

    mujoco.mj_forward(model, sim_data)

def physics_step_to_pose(target_pose, real_dt):

    """

    Advance actual MuJoCo dynamics for approximately one display-frame.

    Gravity, inertia, collision, friction and actuators all participate.

    """

    set_ctrl_from_pose(target_pose)

    sim_dt = float(model.opt.timestep)

    n_steps = max(

        1,

        int(round(real_dt / sim_dt))

    )

    # Prevent a long GUI stall from producing a huge physics jump.

    n_steps = min(n_steps, 100)

    for _ in range(n_steps):

        update_balance(sim_dt)
        set_ctrl_from_pose(target_pose)
        mujoco.mj_step(model, sim_data)

def physics_joint_error(target_pose):

    """

    Maximum absolute actuated-joint tracking error in radians.

    Useful for debugging whether a motor can actually reach a pose.

    """

    errors = []

    for key in JOINT_TO_ACTUATOR:

        q = sim_data.qpos[

            joint_qpos_adr[key]

        ]

        errors.append(

            abs(target_pose[key] - q)

        )

    return max(errors) if errors else 0.0

def sync_working_joints_from_physics(pose):

    """

    Save the ACTUAL simulated joint angles into the working pose.

    Root XYZ is intentionally left as the editor/reference value;

    the free root remains a physics result.

    """

    for key in JOINTS:

        if key in BALANCE_SIGN and key in balance_manual:
            pose[key] = balance_manual[key]
            continue

        pose[key] = float(

            sim_data.qpos[

                joint_qpos_adr[key]

            ]

        )

def physics_edit_step(pose, real_dt):

    """

    Live physics editing:

    mouse/IK changes pose targets -> actuators -> mj_step().

    """

    set_ctrl_from_pose(pose)

    sim_dt = float(model.opt.timestep)

    n_steps = max(1, int(round(real_dt / sim_dt)))

    n_steps = min(n_steps, 100)

    for _ in range(n_steps):

        update_balance(sim_dt)
        set_ctrl_from_pose(pose)
        mujoco.mj_step(model, sim_data)

# ============================================================

# PYGAME

# ============================================================

pygame.init()

screen = pygame.display.set_mode(

    (WINDOW_W, WINDOW_H)

)

pygame.display.set_caption(

    "MuJoCo Whole-Body IK Editor"

)

clock = pygame.time.Clock()

# Increase these two values to enlarge editor text further.

UI_FONT_SIZE = 20

DETAIL_FONT_SIZE = 18

font = pygame.font.SysFont("Arial", UI_FONT_SIZE)

small_font = pygame.font.SysFont("Arial", DETAIL_FONT_SIZE)

# ============================================================

# DRAW GRID

# ============================================================

def draw_grid():

    for x in range(0, WINDOW_W, 50):

        pygame.draw.line(

            screen,

            GRID,

            (x, 0),

            (x, WINDOW_H),

            1

        )

    for y in range(0, WINDOW_H, 50):

        pygame.draw.line(

            screen,

            GRID,

            (0, y),

            (WINDOW_W, y),

            1

        )


# ============================================================

# DRAW ROBOT

# ============================================================

def draw_line(a, b, width=4):

    pygame.draw.line(

        screen,

        LINK_COLOR,

        a.astype(int),

        b.astype(int),

        width

    )

def draw_point(name, pos, color, draggable=False):

    p = pos.astype(int)

    pygame.draw.circle(

        screen,

        color,

        p,

        POINT_RADIUS

    )

    pygame.draw.circle(

        screen,

        WHITE,

        p,

        POINT_RADIUS,

        2

    )

    if draggable:

        pygame.draw.circle(

            screen,

            WHITE,

            p,

            POINT_RADIUS + 6,

            1

        )

    label = small_font.render(

        name,

        True,

        WHITE

    )

    screen.blit(

        label,

        (p[0] + 12, p[1] - 8)

    )

def draw_body_center(pos):

    p = pos.astype(int)

    # 大紅點

    pygame.draw.circle(

        screen,

        RED,

        p,

        BODY_CENTER_RADIUS

    )

    pygame.draw.circle(

        screen,

        WHITE,

        p,

        BODY_CENTER_RADIUS,

        2

    )

    # 顯示可拖動外圈

    pygame.draw.circle(

        screen,

        WHITE,

        p,

        BODY_CENTER_RADIUS + 5,

        1

    )

def draw_robot():

    p = get_skeleton_points()

    # Arms

    draw_line(

        p["L_shoulder"],

        p["L_elbow"]

    )

    draw_line(

        p["L_elbow"],

        p["L_hand"]

    )

    draw_line(

        p["R_shoulder"],

        p["R_elbow"]

    )

    draw_line(

        p["R_elbow"],

        p["R_hand"]

    )

    # Shoulder line

    draw_line(

        p["L_shoulder"],

        p["R_shoulder"],

        5

    )

    # Pelvis roll joints

    draw_point(

        "L pelvis",

        p["L_pelvis_roll"],

        BLUE

    )

    draw_point(

        "R pelvis",

        p["R_pelvis_roll"],

        BLUE

    )

    # ============================================================

    # LEGS

    # pelvis -> pelvis roll -> hip -> knee -> foot

    # ============================================================

    draw_line(p["body_center"], p["L_pelvis_roll"])

    draw_line(p["body_center"], p["R_pelvis_roll"])

    draw_line(p["L_pelvis_roll"], p["L_hip"])

    draw_line(p["R_pelvis_roll"], p["R_hip"])

    draw_line(p["L_hip"], p["L_knee"])

    draw_line(p["R_hip"], p["R_knee"])

    draw_line(p["L_knee"], p["L_foot"])

    draw_line(p["R_knee"], p["R_foot"])

    # Joint points

    draw_point(

        "L shoulder",

        p["L_shoulder"],

        BLUE

    )

    draw_point(

        "R shoulder",

        p["R_shoulder"],

        BLUE

    )

    draw_point(

        "L elbow",

        p["L_elbow"],

        MAGENTA

    )

    draw_point(

        "R elbow",

        p["R_elbow"],

        MAGENTA

    )

    draw_point(

        "L hand",

        p["L_hand"],

        YELLOW,

        True

    )

    draw_point(

        "R hand",

        p["R_hand"],

        YELLOW,

        True

    )

    draw_point(

        "L hip",

        p["L_hip"],

        BLUE, True

    )

    draw_point(

        "R hip",

        p["R_hip"],

        BLUE, True

    )

    draw_point(

        "L knee",

        p["L_knee"],

        MAGENTA

    )

    draw_point(

        "R knee",

        p["R_knee"],

        MAGENTA

    )

    draw_point(

        "L foot",

        p["L_foot"],

        RED,

        True

    )

    draw_point(

        "R foot",

        p["R_foot"],

        RED,

        True

    )

    # Body Center 最後畫，避免被 line 蓋住

    draw_body_center(

        p["body_center"]

    )

# ============================================================

# HIT TEST

# ============================================================

DRAGGABLE = [

    "body_center",

    "L_hand","R_hand",

    "L_hip","R_hip",

    "L_foot","R_foot",

]

selected_point = None

# Body Center drag 專用

body_drag_last_mouse = None

body_fixed_targets = None

def point_hit(mouse, point, radius=HIT_RADIUS):

    mouse = np.array(mouse, dtype=float)

    return (

        np.linalg.norm(mouse - point)

        <= radius

    )

# ============================================================

# SMOOTH STEP

# ============================================================

def smoothstep(t):

    """

    Smooth interpolation from 0 -> 1.

    t = 0 : start

    t = 1 : end

    Start/end velocity are both zero.

    """

    t = float(np.clip(t, 0.0, 1.0))

    return t * t * (3.0 - 2.0 * t)

# ============================================================

# POSE INTERPOLATION

# ============================================================

def interpolate_pose(pose_a, pose_b, alpha):

    """

    alpha:

        0.0 = pose_a

        1.0 = pose_b

    """

    alpha = float(np.clip(alpha, 0.0, 1.0))

    result = {}

    # Root position

    result["root_x"] = (

        (1.0 - alpha) * pose_a["root_x"]

        + alpha * pose_b["root_x"]

    )

    result["root_y"] = (

        (1.0 - alpha) * pose_a["root_y"]

        + alpha * pose_b["root_y"]

    )

    result["root_z"] = (

        (1.0 - alpha) * pose_a["root_z"]

        + alpha * pose_b["root_z"]

    )

    # Joint angles

    for key in JOINTS:

        result[key] = (

            (1.0 - alpha) * pose_a[key]

            + alpha * pose_b[key]

        )

    return result

# ============================================================

# PREVIEW IN MUJOCO

# ============================================================

def preview_pose_in_mujoco(pose):

    sim_data.qpos[:] = model.qpos0

    sim_data.qvel[:] = 0

    # Root translation

    sim_data.qpos[ROOT_X] = pose["root_x"]

    sim_data.qpos[ROOT_Y] = pose["root_y"]

    sim_data.qpos[ROOT_Z] = pose["root_z"]

    # Joints

    for key in JOINTS:

        sim_data.qpos[

            joint_qpos_adr[key]

        ] = pose[key]

    mujoco.mj_forward(

        model,

        sim_data

    )

# ============================================================

# SHOW POSE

# 同時同步 Pygame Editor + MuJoCo Viewer

# ============================================================

def show_pose(pose, debug=True):

    # ----------------------------------------

    # 1. 更新 Pygame IK Editor 使用的 data

    # ----------------------------------------

    apply_pose_to_edit_data(pose)

    # ----------------------------------------

    # 2. 更新 MuJoCo Viewer 使用的 sim_data

    # ----------------------------------------

    if PHYSICS_MODE:

        set_ctrl_from_pose(pose)

    else:

        preview_pose_in_mujoco(pose)

    # ----------------------------------------

    # 3. Debug：確認 Pose 真的不同

    # ----------------------------------------

    if debug:

        print(

            "SHOW POSE |",

            "L shoulder:",

            round(np.degrees(pose["L_shoulder_roll"]), 1),

            "| R shoulder:",

            round(np.degrees(pose["R_shoulder_roll"]), 1),

            "| L elbow:",

            round(np.degrees(pose["L_elbow_roll"]), 1),

            "| R elbow:",

            round(np.degrees(pose["R_elbow_roll"]), 1),

            "| L pelvis:",

            round(np.degrees(pose["L_pelvis_roll"]), 1),

            "| R pelvis:",

            round(np.degrees(pose["R_pelvis_roll"]), 1),

        )

# ============================================================

# UI

# ============================================================

def draw_text_panel(rows, bottom=False):
    """Stack status panels in a separate sidebar, outside the robot view."""
    global status_panel_next_y
    rendered = []
    for text, color, face in rows:
        line = ""
        for word in text.split():
            candidate = f"{line} {word}".strip()
            if line and face.size(candidate)[0] > STATUS_PANEL_W - 30:
                rendered.append((face.render(line, True, color), face.get_linesize() + 4))
                line = word
            else:
                line = candidate
        rendered.append((face.render(line, True, color), face.get_linesize() + 4))
    top = status_panel_next_y if bottom else 0
    height = 16 + sum(h for _, h in rendered)
    pygame.draw.rect(screen, BG, (ROBOT_VIEW_W, top, STATUS_PANEL_W, height))
    y = top + 8
    for surface, advance in rendered:
        screen.blit(surface, (ROBOT_VIEW_W + 15, y))
        y += advance
    status_panel_next_y = top + height


def draw_ui():

    state = (

        "PHYSICS PLAYING" if PHYSICS_PLAYING else

        "PHYSICS EDIT" if PHYSICS_MODE else

        "KINEMATIC PLAYING" if PLAYING else

        "EDIT"

    )

    if current_pose_index == -1:

        label = (

            f"Working Pose | "

            f"Saved: {len(poses)}/{MAX_POSES}"

        )

        pose = working_pose

    else:

        label = (

            f"Viewing Pose "

            f"{current_pose_index + 1}/{len(poses)}"

        )

        pose = poses[current_pose_index]

    rows = [

        (

            f"{label} | {state}",

            WHITE,

            font

        ),

        (

            "N: Save | DEL: Delete | "

            "4/5: Prev/Next | HOME/END: First/Last | "

            "F: Physics | ENTER: Play | R: Reset | J: Fit robot",

            WHITE,

            small_font

        ),

        (

            f"Pelvis L/R: "

            f"{np.degrees(pose['L_pelvis_roll']):.1f} / "

            f"{np.degrees(pose['R_pelvis_roll']):.1f} deg"

            f"   Hip L/R: "

            f"{np.degrees(pose['L_hip_roll']):.1f} / "

            f"{np.degrees(pose['R_hip_roll']):.1f} deg",

            WHITE,

            small_font

        ),

        (
            f"Hip Yaw L/R: "
            f"{np.degrees(pose['L_hip_yaw']):+.1f} / "
            f"{np.degrees(pose['R_hip_yaw']):+.1f} deg",

            YELLOW,

            small_font

        ),

        (

            f"Knee L/R: "

            f"{np.degrees(pose['L_knee_roll']):.1f} / "

            f"{np.degrees(pose['R_knee_roll']):.1f} deg"

            f"   Zoom: {SCALE / 1500.0:.2f}x | "

            f"Preview: {'ON' if LIVE_PREVIEW else 'OFF'}",

            WHITE,

            small_font

        ),

    ]

    if selected_point == "body_center":

        rows.append(

            (

                "WHOLE-BODY IK: 4 endpoints fixed in X/Z",

                RED,

                small_font

            )

        )

    draw_text_panel(rows)

# ============================================================

# MANUAL WRISTS / READ-ONLY ATTITUDE

# ============================================================

WRIST_STEP = np.deg2rad(1.0)

WRIST_KEYS = {

    pygame.K_1: ("L_wrist_pitch", -1),

    pygame.K_KP1: ("L_wrist_pitch", -1),

    pygame.K_2: ("L_wrist_pitch", 1),

    pygame.K_KP2: ("L_wrist_pitch", 1),

    pygame.K_7: ("R_wrist_pitch", 1),

    pygame.K_KP7: ("R_wrist_pitch", 1),

    pygame.K_8: ("R_wrist_pitch", -1),

    pygame.K_KP8: ("R_wrist_pitch", -1),

}

# Each key press changes only the selected manual joint target.
SHOULDER_STEP = np.deg2rad(1.0)
HIP_YAW_STEP = np.deg2rad(1.0)

SHOULDER_KEYS = {
    pygame.K_q: ("L_shoulder_pitch", 1),
    pygame.K_a: ("L_shoulder_pitch", -1),
    pygame.K_w: ("R_shoulder_pitch", 1),
    pygame.K_s: ("R_shoulder_pitch", -1),
}

# Manual hip-yaw control:
#   Left hip yaw  : Z (-1 deg) / X (+1 deg)
#   Right hip yaw : C (-1 deg) / V (+1 deg)
HIP_YAW_KEYS = {
    pygame.K_z: ("L_hip_yaw", -1),
    pygame.K_x: ("L_hip_yaw", 1),
    pygame.K_c: ("R_hip_yaw", -1),
    pygame.K_v: ("R_hip_yaw", 1),
}

MANUAL_JOINT_KEYS = {
    **WRIST_KEYS,
    **SHOULDER_KEYS,
    **HIP_YAW_KEYS,
}


def adjust_manual_joint(pose, key, direction):
    if key in ("L_hip_yaw", "R_hip_yaw"):
        step = HIP_YAW_STEP
    elif key in BALANCE_SIGN:
        step = SHOULDER_STEP
    else:
        step = WRIST_STEP

    pose[key] = clamp_joint(
        key,
        pose[key] + direction * step
    )


def torso_rotation(data):

    return data.xmat[BODY["torso"]].reshape(3, 3).copy()

# Calibrate away the CAD body's initial rotation. The supplied wall lies

# toward world +Y; reference world +Z is upright. This is simulation truth,

# not an IMU fusion estimate. Sensors in XML remain available separately.

attitude_reference = torso_rotation(sim_data)

def relative_attitude(rotation, reference):

    delta = rotation @ reference.T

    up = delta[:, 2]

    back = np.degrees(np.arctan2(-up[1], up[2]))

    side = np.degrees(np.arctan2(up[0], up[2]))

    yaw = np.degrees(np.arctan2(delta[1, 0], delta[0, 0]))

    return back, side, yaw

def draw_manual_status():

    # Recompute on a separate MjData so display never mutates physics state

    # or uses the previous integration step's cached body transforms.

    attitude_data.qpos[:] = sim_data.qpos if PHYSICS_MODE else edit_data.qpos

    mujoco.mj_kinematics(model, attitude_data)

    back, side, yaw = relative_attitude(

        torso_rotation(attitude_data), attitude_reference

    )

    label = "PHYSICS actual" if PHYSICS_MODE else "KINEMATIC preview"

    rows = [

        ("MANUAL WRIST | Left: 1 (-1 deg) / 2 (+1 deg) | Right: 7 (+1 deg) / 8 (-1 deg)", YELLOW, small_font),

        (f"{label} | back(+)/forward(-): {back:+.1f} deg", WHITE, font),

        (f"Side (+X): {side:+.1f} deg | Yaw: {yaw:+.1f} deg | I: Zero tilt", WHITE, small_font),

    ]

    for key, short in (("L_wrist_pitch", "Left"), ("R_wrist_pitch", "Right")):

        if PHYSICS_MODE:

            target = sim_data.ctrl[JOINT_TO_ACTUATOR[key]]

            actual = sim_data.qpos[joint_qpos_adr[key]]

        else:

            target = actual = edit_data.qpos[joint_qpos_adr[key]]

        rows.append((f"{short} wrist pitch | Target: {np.degrees(target):+.1f} deg | Actual: {np.degrees(actual):+.1f} deg", WHITE, small_font))

    rows.append(("Shift+P: Live Preview | Manual joint keys stop playback. Click this window for keys.", GREY, small_font))

    rows.append(("Shoulder manual: L Q (+1 deg) / A (-1 deg) | R W (+1 deg) / S (-1 deg)", YELLOW, small_font))
    rows.append(("Hip yaw manual: L Z (-1 deg) / X (+1 deg) | R C (-1 deg) / V (+1 deg)", YELLOW, small_font))

    for key, short in (("L_hip_yaw", "L"), ("R_hip_yaw", "R")):
        if PHYSICS_MODE and key in JOINT_TO_ACTUATOR:
            target = sim_data.ctrl[JOINT_TO_ACTUATOR[key]]
            actual = sim_data.qpos[joint_qpos_adr[key]]
        else:
            target = actual = edit_data.qpos[joint_qpos_adr[key]]

        rows.append((
            f"{short} hip yaw | Target: {np.degrees(target):+.1f} deg | "
            f"Actual: {np.degrees(actual):+.1f} deg",
            WHITE,
            small_font
        ))

    rows.append((f"Auto Balance: {'ON' if AUTO_BALANCE else 'OFF'} | {SUPPORT_MODE} | B: toggle | O: LEFT / K: BOTH / P: RIGHT", YELLOW, small_font))
    rows.append((f"Target {BALANCE_TARGET_BACK_DEG:+.1f} | Deadband {BALANCE_DEADBAND_DEG:.1f} | Kp {BALANCE_KP:.2f} | Max {BALANCE_MAX_CORRECTION_DEG:.1f} | Rate {BALANCE_RATE_DEG_S:.1f} deg/s", WHITE, small_font))
    rows.append((f"Correction: {balance_correction_deg if PHYSICS_MODE else 0.0:+.2f} deg | BOTH: no correction | Physics only", WHITE, small_font))
    for key, short in (("L_shoulder_pitch", "L"), ("R_shoulder_pitch", "R")):
        manual = balance_manual.get(key, working_pose[key]) if PHYSICS_MODE else edit_data.qpos[joint_qpos_adr[key]]
        command = balance_command.get(key, manual) if PHYSICS_MODE else manual
        rows.append((f"{short} shoulder | Manual {np.degrees(manual):+.2f} | Command {np.degrees(command):+.2f} | Applied {np.degrees(command-manual):+.2f} deg", WHITE, small_font))
    draw_text_panel(rows, bottom=True)

attitude_data = mujoco.MjData(model)

# ============================================================

# VIEWER

# ============================================================

viewer = mujoco.viewer.launch_passive(

    model,

    sim_data

)

# ============================================================

# MAIN LOOP

# ============================================================

fit_robot_view()

running = True

while running and viewer.is_running():

    # ========================================================

    # EVENTS

    # ========================================================

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        # ====================================================

        # MOUSE WHEEL = ZOOM

        # ====================================================

        elif event.type == pygame.MOUSEWHEEL and pygame.mouse.get_pos()[0] < ROBOT_VIEW_W:

            # -----------------------------

            # ZOOM

            # -----------------------------

            if event.y > 0:

                SCALE *= ZOOM_FACTOR

            elif event.y < 0:

                SCALE /= ZOOM_FACTOR

            SCALE = float(

                np.clip(

                    SCALE,

                    MIN_SCALE,

                    MAX_SCALE

                )

            )

            # -----------------------------

            # RECENTER ROBOT

            # -----------------------------

            camera_world_center = (

                get_skeleton_world_center()

            )

            print(

                f"EDITOR ZOOM: {SCALE:.0f}"

            )

        # ====================================================

        # KEYBOARD

        # ====================================================

        elif event.type == pygame.KEYDOWN:

            print(

                "KEY RECEIVED:",

                pygame.key.name(event.key),

                "code =",

                event.key

            )

            # ESC = EXIT

            if event.key == pygame.K_ESCAPE:

                running = False

            # R = RESET ROBOT TO INITIAL STARTUP POSE
            # Saved poses are preserved.
            elif event.key == pygame.K_r:
                PLAYING = False
                PHYSICS_PLAYING = False
                PHYSICS_EDITING = PHYSICS_MODE

                current_pose_index = -1
                selected_point = None
                body_drag_last_mouse = None
                body_fixed_targets = None

                # Restore the editable target pose.
                working_pose = copy.deepcopy(initial_pose)
                reset_balance()

                if PHYSICS_MODE:
                    # Reset qpos/qvel/control so gravity/contact simulation
                    # starts again from exactly the initial pose.
                    reset_physics_to_initial()
                    apply_pose_to_edit_data(working_pose)
                else:
                    # Reset both Pygame editor and MuJoCo preview.
                    show_pose(working_pose, debug=False)

                fit_robot_view()
                print("RESET: robot returned to initial pose")

            elif event.key == pygame.K_j:
                fit_robot_view()

            elif event.key == pygame.K_b:
                AUTO_BALANCE = not AUTO_BALANCE
                reset_balance()
                if PHYSICS_MODE:
                    set_ctrl_from_pose(balance_manual_pose())
                print("Auto Balance:", AUTO_BALANCE)

            elif event.key in (pygame.K_o, pygame.K_k, pygame.K_p) and not (event.key == pygame.K_p and event.mod & pygame.KMOD_SHIFT):
                SUPPORT_MODE = {pygame.K_o: "LEFT_SUPPORT", pygame.K_k: "BOTH",
                                pygame.K_p: "RIGHT_SUPPORT"}[event.key]
                reset_balance()
                if PHYSICS_MODE:
                    set_ctrl_from_pose(balance_manual_pose())
                print("Support:", SUPPORT_MODE)

            elif event.key in MANUAL_JOINT_KEYS:

                if PHYSICS_MODE:

                    # Preserve every current motor command when taking over.

                    for key, aid in JOINT_TO_ACTUATOR.items():

                        working_pose[key] = balance_manual.get(key, float(sim_data.ctrl[aid]))

                elif PLAYING:

                    for key in JOINTS:

                        working_pose[key] = float(edit_data.qpos[joint_qpos_adr[key]])

                    for key, adr in (("root_x", ROOT_X), ("root_y", ROOT_Y), ("root_z", ROOT_Z)):

                        working_pose[key] = float(edit_data.qpos[adr])

                elif current_pose_index != -1:

                    working_pose = copy.deepcopy(poses[current_pose_index])

                PLAYING = False

                PHYSICS_PLAYING = False

                PHYSICS_EDITING = PHYSICS_MODE

                current_pose_index = -1

                selected_point = None

                body_drag_last_mouse = None

                body_fixed_targets = None

                key, direction = MANUAL_JOINT_KEYS[event.key]

                adjust_manual_joint(working_pose, key, direction)

                apply_pose_to_edit_data(working_pose)

                if PHYSICS_MODE:

                    set_ctrl_from_pose(working_pose)

                elif LIVE_PREVIEW:

                    preview_pose_in_mujoco(working_pose)

                print(f"MANUAL {key}: {np.degrees(working_pose[key]):+.1f} deg target")

            elif event.key == pygame.K_i:

                attitude_data.qpos[:] = sim_data.qpos if PHYSICS_MODE else edit_data.qpos

                mujoco.mj_kinematics(model, attitude_data)

                attitude_reference = torso_rotation(attitude_data)
                reset_balance()

                print("Tilt display zeroed. Motor targets unchanged.")

            # ------------------------------------------------

            # N = SAVE CURRENT WORKING POSE

            # ------------------------------------------------

            elif event.key == pygame.K_n:

                PLAYING = False

                PHYSICS_PLAYING = False

                # Physics mode:

                # 儲存 MuJoCo 真正到達的 joint angle

                if PHYSICS_MODE:

                    sync_working_joints_from_physics(

                        working_pose

                    )

                    apply_pose_to_edit_data(

                        working_pose

                    )

                # 回到 Working Pose 狀態

                current_pose_index = -1

                # 最多 100 poses

                if len(poses) >= MAX_POSES:

                    print(

                        f"Cannot save: maximum "

                        f"{MAX_POSES} poses reached."

                    )

                else:

                    # 儲存完整目前姿勢

                    poses.append(

                        copy.deepcopy(working_pose)

                    )

                    # 儲存後保持目前姿勢

                    show_pose(

                        working_pose,

                        debug=False

                    )

                    print(

                        f"SAVED POSE "

                        f"{len(poses)}/{MAX_POSES} | "

                        f"Continue editing from this pose."

        )

            # ------------------------------------------------

            # DELETE = DELETE CURRENT VIEWED POSE

            # ------------------------------------------------

            elif event.key == pygame.K_DELETE:

                PLAYING = False

                PHYSICS_PLAYING = False

                if len(poses) == 0:

                    print("No saved poses to delete.")

                elif current_pose_index == -1:

                    # Working Pose 狀態下，刪除最後一個

                    deleted_number = len(poses)

                    poses.pop()

                    print(

                        f"DELETED POSE {deleted_number} | "

                        f"{len(poses)} poses remaining."

                    )

                    show_pose(

                        working_pose,

                        debug=False

                    )

                else:

                    deleted_number = current_pose_index + 1

                    poses.pop(current_pose_index)

                    print(

                        f"DELETED POSE {deleted_number} | "

                        f"{len(poses)} poses remaining."

                    )

                    # 刪除後回 Working Pose

                    current_pose_index = -1

                    show_pose(

                        working_pose,

                        debug=False

                    )

            # ------------------------------------------------

            # 4 = VIEW PREVIOUS SAVED POSE

            # ------------------------------------------------

            elif event.key in (

                pygame.K_4,

                pygame.K_KP4

            ):

                PLAYING = False

                if len(poses) == 0:

                    print("No saved poses yet.")

                else:

                    if current_pose_index == -1:

                        current_pose_index = len(poses) - 1

                    else:

                        current_pose_index = max(

                            0,

                            current_pose_index - 1

                        )

                    show_pose(

                        poses[current_pose_index]

                    )

                    print(

                        f"<<< VIEW POSE "

                        f"{current_pose_index + 1}/{len(poses)}"

                    )

            # ------------------------------------------------

            # 5 = VIEW NEXT SAVED POSE

            #

            # 從最後一個 saved pose 再按 5，回到 working pose。

            # ------------------------------------------------

            elif event.key in (

                pygame.K_5,

                pygame.K_KP5

            ):

                PLAYING = False

                if len(poses) == 0:

                    print("No saved poses yet.")

                elif current_pose_index == -1:

                    # Already at working pose.

                    show_pose(working_pose)

                    print("WORKING POSE")

                elif current_pose_index < len(poses) - 1:

                    current_pose_index += 1

                    show_pose(

                        poses[current_pose_index]

                    )

                    print(

                        f">>> VIEW POSE "

                        f"{current_pose_index + 1}/{len(poses)}"

                    )

                else:

                    current_pose_index = -1

                    show_pose(

                        working_pose

                    )

                    print(">>> WORKING POSE")

            # ------------------------------------------------

            # SPACE = RETURN TO WORKING POSE

            # ------------------------------------------------

            elif event.key == pygame.K_SPACE:

                PLAYING = False

                current_pose_index = -1

                show_pose(

                    working_pose

                )

                print("RETURN TO WORKING POSE")

            # ------------------------------------------------

            # F = TOGGLE LIVE PHYSICS EDIT MODE

            #

            # ON:

            #   - current working pose becomes actuator target

            #   - gravity/contact stay active

            #   - mouse can still move joints

            #   - N saves the current physics pose

            #

            # OFF:

            #   - return to original kinematic editor

            # ------------------------------------------------

            elif event.key == pygame.K_f:

                PLAYING = False

                PHYSICS_PLAYING = False

                PHYSICS_MODE = not PHYSICS_MODE

                current_pose_index = -1

                if PHYSICS_MODE:

                    PHYSICS_EDITING = True

                    # Start physics from the CURRENT working pose.

                    reset_balance()

                    mujoco.mj_resetData(model, sim_data)

                    sim_data.qpos[:] = model.qpos0

                    sim_data.qvel[:] = 0

                    sim_data.qpos[ROOT_X] = working_pose["root_x"]

                    sim_data.qpos[ROOT_Y] = working_pose["root_y"]

                    sim_data.qpos[ROOT_Z] = working_pose["root_z"]

                    for key in JOINTS:

                        sim_data.qpos[

                            joint_qpos_adr[key]

                        ] = working_pose[key]

                    set_ctrl_from_pose(working_pose)

                    mujoco.mj_forward(model, sim_data)

                    # ============================================================

                    # SAVE RL INITIAL STATE

                    # 這是「剛按 F、還沒開始掉落」的狀態

                    # ============================================================

                    RL_QPOS_PATH = os.path.join(BASE_DIR, "hang_start_qpos.npy")

                    RL_CTRL_PATH = os.path.join(BASE_DIR, "hang_start_ctrl.npy")

                    np.save(

                        RL_QPOS_PATH,

                        sim_data.qpos.copy()

                    )

                    np.save(

                        RL_CTRL_PATH,

                        sim_data.ctrl.copy()

                    )

                    print("\n========================================")

                    print("RL INITIAL STATE SAVED")

                    print("========================================")

                    print("qpos:")

                    print(sim_data.qpos)

                    print()

                    print("ctrl:")

                    print(sim_data.ctrl)

                    print()

                    print("qpos file:", RL_QPOS_PATH)

                    print("ctrl file:", RL_CTRL_PATH)

                    print("nq =", model.nq)

                    print("nu =", model.nu)

                    print("========================================\n")

                    print(

                        "PHYSICS EDIT MODE: ON | "

                        "Move joints -> N save -> move -> N save -> ENTER"

                    )

                else:

                    PHYSICS_EDITING = False

                    preview_pose_in_mujoco(working_pose)

                    print(

                        "PHYSICS EDIT MODE: OFF | "

                        "Back to kinematic editor"

                    )

            # ------------------------------------------------

            # ENTER = PLAY:

            # INITIAL -> POSE 1 -> POSE 2 -> POSE 3 -> ...

            # ------------------------------------------------

            elif event.key in (

                pygame.K_RETURN,

                pygame.K_KP_ENTER

            ):

                if len(poses) == 0:

                    print(

                        "No saved poses. "

                        "Move robot and press N to save Pose 1."

                    )

                else:

                    selected_point = None

                    body_drag_last_mouse = None

                    body_fixed_targets = None

                    current_pose_index = -1

                    if PHYSICS_MODE:

                        PLAYING = False

                        PHYSICS_PLAYING = True

                        PHYSICS_EDITING = False

                        physics_segment = 0

                        physics_segment_t = 0.0

                        # qpos is initialized only here.

                        # From this point on, physics owns the root/body motion.

                        reset_physics_to_initial()

                        apply_pose_to_edit_data(initial_pose)

                        print(

                            f"PHYSICS PLAY: INITIAL -> "

                            f"POSE 1 ... POSE {len(poses)}"

                        )

                    else:

                        PHYSICS_PLAYING = False

                        play_segment = 0

                        play_t = 0.0

                        PLAYING = True

                        # Kinematic preview starts from initial pose.

                        show_pose(

                            initial_pose,

                            debug=False

                        )

                        print(

                            f"KINEMATIC PLAY: INITIAL -> "

                            f"POSE 1 ... POSE {len(poses)}"

                        )

            # Shift+P = LIVE PREVIEW ON / OFF

            elif event.key == pygame.K_p and event.mod & pygame.KMOD_SHIFT:

                LIVE_PREVIEW = not LIVE_PREVIEW

                print(

                    "Live MuJoCo Preview:",

                    "ON" if LIVE_PREVIEW else "OFF"

                )

                if LIVE_PREVIEW:

                    if current_pose_index == -1:

                        show_pose(

                            working_pose,

                            debug=False

                        )

                    else:

                        show_pose(

                            poses[current_pose_index],

                            debug=False

                        )

            # ------------------------------------------------

            # R = RESET WORKING POSE TO INITIAL

            # 已儲存 poses 不刪除

            # ------------------------------------------------

            elif event.key == pygame.K_r:

                PLAYING = False

                current_pose_index = -1

                working_pose = copy.deepcopy(

                    initial_pose

                )

                if PHYSICS_MODE:

                    reset_physics_to_initial()

                    PHYSICS_EDITING = True

                    apply_pose_to_edit_data(working_pose)

                else:

                    show_pose(

                        working_pose

                    )

                print(

                    "WORKING POSE RESET TO INITIAL "

                    "(saved poses kept)"

                )

            # ------------------------------------------------

            # HOME = VIEW POSE 1

            # ------------------------------------------------

            elif event.key == pygame.K_HOME:

                PLAYING = False

                PHYSICS_PLAYING = False

                if len(poses) > 0:

                    current_pose_index = 0

                    show_pose(

                        poses[current_pose_index],

                        debug=False

                    )

                    print(

                        f"VIEW POSE 1/{len(poses)}"

                    )

                else:

                    print("No saved poses.")

            # ------------------------------------------------

            # END = VIEW LAST POSE

            # ------------------------------------------------

            elif event.key == pygame.K_END:

                PLAYING = False

                PHYSICS_PLAYING = False

                if len(poses) > 0:

                    current_pose_index = len(poses) - 1

                    show_pose(

                        poses[current_pose_index],

                        debug=False

                    )

                    print(

                        f"VIEW LAST POSE "

                        f"{current_pose_index + 1}/{len(poses)}"

                    )

                else:

                    print("No saved poses.")

        # ====================================================

        # MOUSE DOWN

        # Only working pose can be edited.

        # If viewing a saved pose, clicking returns to working.

        # ====================================================

        elif (

            event.type == pygame.MOUSEBUTTONDOWN

            and event.button == 1
            and event.pos[0] < ROBOT_VIEW_W

            and not PLAYING

            and not PHYSICS_PLAYING

        ):

            if current_pose_index != -1:

                current_pose_index = -1

                show_pose(

                    working_pose,

                    debug=False

                )

            p = get_skeleton_points()

            if point_hit(

                event.pos,

                p["body_center"],

                BODY_CENTER_RADIUS + 8

            ):

                selected_point = "body_center"

                body_fixed_targets = (

                    get_end_effector_targets()

                )

                body_drag_last_mouse = np.array(

                    event.pos,

                    dtype=float

                )

                print("Body Center selected.")

            else:

                for name in [

                    "L_hand", "R_hand",

                    "L_hip", "R_hip",

                    "L_foot", "R_foot"

                ]:

                    if point_hit(

                        event.pos,

                        p[name]

                    ):

                        selected_point = name

                        break

        # ====================================================

        # MOUSE UP

        # ====================================================

        elif (

            event.type == pygame.MOUSEBUTTONUP

            and event.button == 1
            and event.pos[0] < ROBOT_VIEW_W

        ):

            selected_point = None

            body_drag_last_mouse = None

            body_fixed_targets = None

        # ====================================================

        # MOUSE DRAG = EDIT working_pose ONLY

        # ====================================================

        elif (

            event.type == pygame.MOUSEMOTION

            and selected_point is not None

            and not PLAYING

            and not PHYSICS_PLAYING

        ):

            current_pose_index = -1

            pose = working_pose

            # BODY CENTER DRAG

            if selected_point == "body_center":

                mouse = np.array(

                    event.pos,

                    dtype=float

                )

                delta = (

                    mouse

                    - body_drag_last_mouse

                )

                dx, dz = screen_delta_to_world(

                    delta[0],

                    delta[1]

                )

                pose["root_x"] += dx

                pose["root_z"] += dz

                solve_whole_body(

                    pose,

                    body_fixed_targets

                )

                body_drag_last_mouse = mouse

            # HIP / HAND / FOOT

            else:

                if selected_point in (

                    "L_hip",

                    "R_hip"

                ):

                    solve_hip_drag(

                        selected_point,

                        event.pos,

                        pose

                    )

                else:

                    solve_ik(

                        selected_point,

                        event.pos,

                        pose

                    )

    # ========================================================

    # POSE ANIMATION

    #

    # Segment 0:

    #     initial_pose -> poses[0]

    #

    # Segment 1:

    #     poses[0] -> poses[1]

    #

    # Segment 2:

    #     poses[1] -> poses[2]

    # ========================================================

    dt = clock.get_time() / 1000.0

    if PHYSICS_PLAYING:

        if physics_segment >= len(poses):

            PHYSICS_PLAYING = False

            PHYSICS_EDITING = True

            print("Physics animation finished. Physics editing resumed.")

        else:

            target_pose = poses[physics_segment]

            physics_step_to_pose(

                target_pose,

                dt

            )

            # 2D editor shows the reference target pose.

            # 3D MuJoCo viewer shows the REAL simulated robot.

            apply_pose_to_edit_data(

                target_pose

            )

            physics_segment_t += dt

            if physics_segment_t >= PHYSICS_POSE_DURATION:

                err = physics_joint_error(

                    target_pose

                )

                print(

                    f"Physics reached time for Pose "

                    f"{physics_segment + 1}/{len(poses)} | "

                    f"max joint error = "

                    f"{np.degrees(err):.2f} deg | "

                    f"root z = {sim_data.qpos[ROOT_Z]:.4f} m"

                )

                physics_segment += 1

                physics_segment_t = 0.0

                if physics_segment >= len(poses):

                    PHYSICS_PLAYING = False

                    PHYSICS_EDITING = True

                    # Use the final simulated joint angles as the next

                    # editable starting point.

                    sync_working_joints_from_physics(

                        working_pose

                    )

                    apply_pose_to_edit_data(

                        working_pose

                    )

                    print(

                        "PHYSICS PLAYBACK FINISHED | "

                        f"root z = "

                        f"{sim_data.qpos[ROOT_Z]:.4f} m | "

                        "Physics editing resumed."

                    )

    elif PLAYING:

        if play_segment >= len(poses):

            PLAYING = False

        else:

            if play_segment == 0:

                pose_a = initial_pose

                pose_b = poses[0]

            else:

                pose_a = poses[play_segment - 1]

                pose_b = poses[play_segment]

            play_t += dt / POSE_DURATION

            raw_alpha = min(

                play_t,

                1.0

            )

            alpha = smoothstep(

                raw_alpha

            )

            animation_pose = interpolate_pose(

                pose_a,

                pose_b,

                alpha

            )

            # Pygame 2D + MuJoCo 3D synchronized.

            apply_pose_to_edit_data(

                animation_pose

            )

            preview_pose_in_mujoco(

                animation_pose

            )

            if play_t >= 1.0:

                show_pose(

                    pose_b,

                    debug=False

                )

                print(

                    f"Reached Pose "

                    f"{play_segment + 1}/{len(poses)}"

                )

                play_segment += 1

                play_t = 0.0

                if play_segment >= len(poses):

                    PLAYING = False

                    # Playback ends at the last saved pose.

                    show_pose(

                        poses[-1],

                        debug=False

                    )

                    # Keep working_pose as the latest edited pose.

                    current_pose_index = -1

                    print(

                        "Animation finished. "

                        "Final saved pose reached."

                    )

    # ========================================================

    # NORMAL PREVIEW

    # ========================================================

    elif PHYSICS_MODE and PHYSICS_EDITING:

        # LIVE PHYSICS EDIT:

        # working_pose is the actuator target.

        # sim_data is the actual robot under gravity/contact.

        physics_edit_step(

            working_pose,

            dt

        )

    elif LIVE_PREVIEW and not PHYSICS_MODE:

        if current_pose_index == -1:

            preview_pose_in_mujoco(

                working_pose

            )

        else:

            preview_pose_in_mujoco(

                poses[current_pose_index]

            )

    # ========================================================

    # MUJOCO VIEWER

    # ========================================================

    viewer.sync()

    # ========================================================

    # DRAW PYGAME IK EDITOR

    # ========================================================

    screen.fill(BG)

    screen.set_clip(pygame.Rect(0, 0, ROBOT_VIEW_W, WINDOW_H))
    draw_grid()


    draw_robot()
    screen.set_clip(None)
    pygame.draw.line(screen, GREY, (ROBOT_VIEW_W, 0), (ROBOT_VIEW_W, WINDOW_H), 2)

    draw_ui()

    draw_manual_status()

    pygame.display.flip()

    clock.tick(FPS)

# ============================================================

# CLOSE

# ============================================================

viewer.close()

pygame.quit()

