from controller import Robot
import pinocchio as pin
import numpy as np
import os
import math

def main():
    print("\n" + "="*50)
    print("🤖 具身智能测试：开环步态行走 (企鹅摇摆步)")
    print("="*50)

    # 1. 唤醒小脑 (Pinocchio)
    urdf_path = os.path.expanduser("~/nao_VLM/nao_robot/nao_description/urdf/naoV50_generated_urdf/nao.urdf")
    if not os.path.exists(urdf_path):
        urdf_path = os.path.expanduser("~/nao_VLM/nao_robot/nao_description/urdf/nao.urdf")
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data = model.createData()
    q_pin = pin.neutral(model)

    # 2. 唤醒肉体 (Webots)
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # 这次我们需要加入控制左右摇摆的 Roll (横滚角) 电机
    tracked_joints = [
        'LHipPitch', 'LKneePitch', 'LAnklePitch', 
        'RHipPitch', 'RKneePitch', 'RAnklePitch',
        'LHipRoll', 'RHipRoll', 'LAnkleRoll', 'RAnkleRoll', # 新增：用于左右重心转移
        'LShoulderPitch', 'RShoulderPitch'
    ]
    
    motors = {}
    sensors = {}
    joint_q_idx_map = {}

    print(">>> 正在绑定行走所需的所有电机...")
    for name in tracked_joints:
        motors[name] = robot.getDevice(name)
        sensor_name = name + 'S'
        sensors[name] = robot.getDevice(sensor_name)
        if sensors[name]:
            sensors[name].enable(timestep)
        if name in model.names:
            joint_id = model.getJointId(name)
            joint_q_idx_map[name] = model.joints[joint_id].idx_q

    # 3. 初始下蹲准备 (降低重心，走路更稳)
    print(">>> 降低重心准备起步...")
    base_knee = 0.5
    base_hip_ankle = -0.25
    for leg in ['L', 'R']:
        motors[f'{leg}KneePitch'].setPosition(base_knee)
        motors[f'{leg}HipPitch'].setPosition(base_hip_ankle)
        motors[f'{leg}AnklePitch'].setPosition(base_hip_ankle)
    
    # 给它 1 秒钟时间稳住重心
    for _ in range(int(1000 / timestep)):
        robot.step(timestep)

    # 4. 行走主循环
    step_count = 0
    print(">>> 开始摇摆行走！")
    
    while robot.step(timestep) != -1:
        step_count += 1
        t = step_count * (timestep / 1000.0)
        
        # 行走频率：每秒完成半步
        freq = 0.8 
        phase = 2 * math.pi * freq * t

        # ====================================================
        # 核心步态算法 (开环 CPG)
        # ====================================================
        
        # 1. 左右重心摇摆 (Roll) -> 决定了机器人会不会摔倒
        # 让躯干左右倾斜，同时脚踝反向补偿，保证脚底板贴地
        sway_amp = 0.15 
        sway = sway_amp * math.sin(phase)
        motors['LHipRoll'].setPosition(sway)
        motors['RHipRoll'].setPosition(sway)
        motors['LAnkleRoll'].setPosition(-sway)
        motors['RAnkleRoll'].setPosition(-sway)

        # 2. 抬腿与前迈 (Pitch) -> 趁着重心在另一边时，抬起空载的腿
        # 当 sway > 0 时，重心偏左，此时抬起右腿 (利用 max 提取半波)
        lift_amp = 0.25
        forward_step = 0.1 # 往前迈的步幅
        
        r_lift = lift_amp * max(0, -math.sin(phase))
        l_lift = lift_amp * max(0, math.sin(phase))
        
        # 叠加基础下蹲角度和迈步角度
        motors['LKneePitch'].setPosition(base_knee + l_lift * 2)
        motors['LHipPitch'].setPosition(base_hip_ankle - l_lift - forward_step * max(0, math.sin(phase)))
        motors['LAnklePitch'].setPosition(base_hip_ankle - l_lift)

        motors['RKneePitch'].setPosition(base_knee + r_lift * 2)
        motors['RHipPitch'].setPosition(base_hip_ankle - r_lift - forward_step * max(0, -math.sin(phase)))
        motors['RAnklePitch'].setPosition(base_hip_ankle - r_lift)
        
        # 双手稍微抬起保持平衡
        motors['LShoulderPitch'].setPosition(1.2)
        motors['RShoulderPitch'].setPosition(1.2)

        # 小脑监控：打印质心 X(前后) Y(左右) 偏移
        if step_count % 20 == 0:
            for name in tracked_joints:
                if name in sensors and name in joint_q_idx_map:
                    q_pin[joint_q_idx_map[name]] = sensors[name].getValue()
            pin.centerOfMass(model, data, q_pin)
            com = data.com[0]
            print(f"Time: {t:.1f}s | CoM Sway(Y): {com[1]:+.4f} | Forward(X): {com[0]:+.4f}")

if __name__ == '__main__':
    main()