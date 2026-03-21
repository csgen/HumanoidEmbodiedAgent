from controller import Supervisor
import pinocchio as pin
import numpy as np
import os
import math
import cv2  # 用于保存摄像头画面

class NaoVlmAPI:
    def __init__(self, robot, pin_model, pin_data, motors, cameras):
        self.robot = robot
        self.model = pin_model
        self.data = pin_data
        self.motors = motors
        self.cameras = cameras
        
        self.hand_frames = {
            "left": self.model.getFrameId("l_wrist") if self.model.existFrame("l_wrist") else self.model.getFrameId("LHand"),
            "right": self.model.getFrameId("r_wrist") if self.model.existFrame("r_wrist") else self.model.getFrameId("RHand")
        }
        self.current_posture = "stand"
        
    # 1. 状态获取
    def get_robot_state(self):
        state = {
            "posture": self.current_posture,
            "head_pitch": self.motors['HeadPitch'].getTargetPosition() if 'HeadPitch' in self.motors else 0.0,
            "head_yaw": self.motors['HeadYaw'].getTargetPosition() if 'HeadYaw' in self.motors else 0.0
        }
        return str(state)

    # 2. 视觉触发
    def capture_camera_image(self, save_path="vlm_view.jpg"):
        if 'CameraTop' not in self.cameras:
            return "ERROR: Top camera not found."
            
        cam = self.cameras['CameraTop']
        image_data = cam.getImage()
        if image_data:
            width, height = cam.getWidth(), cam.getHeight()
            img = np.frombuffer(image_data, np.uint8).reshape((height, width, 4))
            img_bgr = img[:, :, :3]
            cv2.imwrite(save_path, img_bgr)
            return f"SUCCESS: Image saved to {save_path}"
        return "ERROR: Failed to capture image."

    # 3. 转头
    def look_at(self, yaw_angle, pitch_angle):
        yaw = np.clip(yaw_angle, -2.0, 2.0)
        pitch = np.clip(pitch_angle, -0.6, 0.5)
        if 'HeadYaw' in self.motors: self.motors['HeadYaw'].setPosition(yaw)
        if 'HeadPitch' in self.motors: self.motors['HeadPitch'].setPosition(pitch)
        return f"SUCCESS: Look at yaw={yaw:.2f}, pitch={pitch:.2f}"

    # 4. 手臂运动 (IK)
    def move_arm(self, side, x, y, z):
        if side not in ["left", "right"]: return "ERROR: Invalid side."
        target_pos = np.array([x, y, z])
        frame_id = self.hand_frames[side]
        q = getattr(self, 'q_current', pin.neutral(self.model)).copy()
        
        success = False
        for _ in range(20):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current_pos = self.data.oMf[frame_id].translation
            err = target_pos - current_pos
            if np.linalg.norm(err) < 0.005:
                success = True
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
            
            # --------------------------------------------------
            # 【核心修复】：锁住虚拟底盘！
            # 强制将前 6 个自由度（浮动基座）的雅可比矩阵列清零。
            # 这逼迫数学引擎只能通过改变后续的手臂关节来消除误差。
            # --------------------------------------------------
            J[:, :6] = 0.0 
            
            dq = np.linalg.pinv(J) @ err
            q = pin.integrate(self.model, q, dq * 0.5)
            
        if success:
            # 真实下发给 Webots 电机 (带有物理限位安全剪裁)
            for motor_name, motor in self.motors.items():
                if motor_name in self.model.names:
                    joint_id = self.model.getJointId(motor_name)
                    q_idx = self.model.joints[joint_id].idx_q
                    target_angle = float(q[q_idx])
                    
                    # 读取 Webots 里的物理极限边界
                    min_pos = motor.getMinPosition()
                    max_pos = motor.getMaxPosition()
                    
                    # 如果 min 和 max 不相等，说明电机有物理限位，我们就用 numpy 强行剪裁
                    if min_pos != max_pos:
                        target_angle = np.clip(target_angle, min_pos, max_pos)
                        
                    motor.setPosition(target_angle)
            return f"SUCCESS: {side} arm moved to ({x}, {y}, {z})."
        return "WARNING: Target unreachable."

    # 5. 手指开合
    def operate_gripper(self, side, action):
        prefix = 'L' if side == "left" else 'R'
        target_angle = 0.0 if action == "close" else 1.0 
        success_count = 0
        for i in range(1, 9):
            phalanx_name = f"{prefix}Phalanx{i}"
            motor = self.robot.getDevice(phalanx_name)
            if motor:
                motor.setPosition(target_angle)
                success_count += 1
        if success_count > 0:
            return f"SUCCESS: {side} hand {action}ed."
        return f"ERROR: Finger motors not found."

    # 6. 全身姿态
    def set_posture(self, posture_name):
        self.current_posture = posture_name
        val_knee = 1.0 if posture_name == "squat" else 0.0
        val_hip_ankle = -0.5 if posture_name == "squat" else 0.0
            
        for leg in ['L', 'R']:
            if f'{leg}KneePitch' in self.motors: self.motors[f'{leg}KneePitch'].setPosition(val_knee)
            if f'{leg}HipPitch' in self.motors: self.motors[f'{leg}HipPitch'].setPosition(val_hip_ankle)
            if f'{leg}AnklePitch' in self.motors: self.motors[f'{leg}AnklePitch'].setPosition(val_hip_ankle)
        return f"SUCCESS: Posture set to {posture_name}."

    # ==========================================
    # 7. 宏观移动 (Supervisor 平滑滑行)
    # ==========================================
    def navigate_to(self, delta_x, delta_y, delta_theta):
        """
        供 VLM 调用：相对当前位置平滑移动
        """
        # 获取上帝视角的机器人节点
        robot_node = self.robot.getSelf()
        if robot_node is None:
            return "ERROR: Please set 'supervisor' to TRUE in Webots NAO node."

        trans_field = robot_node.getField("translation")
        rot_field = robot_node.getField("rotation") # 如果需要转向可以操作这个

        current_trans = trans_field.getSFVec3f()
        
        print(f">>> [底层导航] 正在平滑移动 X:{delta_x}m, Y:{delta_y}m...")
        
        # 将移动过程切分为 50 帧，看起来像平滑滑行
        frames = 50
        timestep = int(self.robot.getBasicTimeStep())
        
        for i in range(1, frames + 1):
            interp_x = current_trans[0] + (delta_x * i / frames)
            interp_y = current_trans[1] + (delta_y * i / frames)
            # 保持 Z 轴（高度）不变
            trans_field.setSFVec3f([interp_x, interp_y, current_trans[2]])
            
            # 推进仿真时间
            if self.robot.step(timestep) == -1:
                break
                
        return f"SUCCESS: Navigated to relative ({delta_x}, {delta_y})."

    # 8. 说话
    def speak(self, text):
        print(f"\n📢 [NAO]: {text}")
        return "SUCCESS: Text spoken."


def main():
    print("\n" + "="*50)
    print("🤖 具身智能 API：全面自动化阅兵测试")
    print("="*50)

    # 1. 小脑初始化
    urdf_path = os.path.expanduser("~/nao_VLM/nao_robot/nao_description/urdf/naoV50_generated_urdf/nao.urdf")
    if not os.path.exists(urdf_path):
        urdf_path = os.path.expanduser("~/nao_VLM/nao_robot/nao_description/urdf/nao.urdf")
    try:
        model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        data = model.createData()
        q_pin = pin.neutral(model)
    except Exception as e:
        print(f"Pinocchio 加载失败: {e}")
        return

    # 2. 肉体初始化
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    
    tracked_joints = [
        'HeadYaw', 'HeadPitch',
        'LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll', 'LWristYaw',
        'RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll', 'RWristYaw',
        'LHipPitch', 'LKneePitch', 'LAnklePitch',
        'RHipPitch', 'RKneePitch', 'RAnklePitch'
    ]
    
    motors, sensors, joint_q_idx_map, cameras = {}, {}, {}, {}
    
    camera_top = robot.getDevice('CameraTop')
    if camera_top:
        camera_top.enable(timestep)
        cameras['CameraTop'] = camera_top

    # 安全绑定
    for name in tracked_joints:
        motor = robot.getDevice(name)
        if motor: motors[name] = motor
        
        sensor = robot.getDevice(name + 'S')
        if sensor:
            sensor.enable(timestep)
            sensors[name] = sensor
            
        if name in model.names:
            joint_q_idx_map[name] = model.joints[model.getJointId(name)].idx_q

    # 3. 接口实例化
    vlm_api = NaoVlmAPI(robot, model, data, motors, cameras)

    # 4. 自动化测试时间轴
    step_count = 0
    print("\n>>> 开始 API 轮询测试...")
    
    while robot.step(timestep) != -1:
        step_count += 1
        
        # 实时同步物理数据给数学小脑
        for name in tracked_joints:
            if name in sensors and name in joint_q_idx_map:
                q_pin[joint_q_idx_map[name]] = sensors[name].getValue()
        vlm_api.q_current = q_pin 
        
        # ----------- 阅兵时间轴 -----------
        if step_count == 20:
            print("\n[测试 1/8: 语音表达]")
            print(vlm_api.speak("开始 API 全面阅兵！"))
            
        elif step_count == 80:
            print("\n[测试 2/8: 视觉注意力 (看右上方)]")
            print(vlm_api.look_at(yaw_angle=-0.5, pitch_angle=-0.3))
            
        elif step_count == 140:
            print("\n[测试 3/8: 拍照感知]")
            print(vlm_api.capture_camera_image("test_vision.jpg"))
            
        elif step_count == 200:
            print("\n[测试 4/8: 改变全身姿态 (下蹲)]")
            print(vlm_api.set_posture("squat"))
            
        elif step_count == 280:
            print("\n[测试 4.5: 改变全身姿态 (起立)]")
            print(vlm_api.set_posture("stand"))
            
        elif step_count == 360:
            print("\n[测试 5/8: 右手空间运动 (IK 到体前)]")
            print(vlm_api.move_arm("right", x=0.05, y=-0.05, z=-0.05))
            
        elif step_count == 440:
            print("\n[测试 6/8: 手指操作 (闭合)]")
            print(vlm_api.operate_gripper("right", "close"))
            
        elif step_count == 520:
            print("\n[测试 6.5: 手指操作 (张开)]")
            print(vlm_api.operate_gripper("right", "open"))
            
        elif step_count == 600:
            print("\n[测试 7/8: 获取最新状态]")
            print(vlm_api.get_robot_state())
            
        elif step_count == 680:
            print("\n[测试 8/8: 宏观导航 (底层系统 Mock)]")
            print(vlm_api.navigate_to(0.5, 0.0, 0.0))
            
        elif step_count == 760:
            print("\n🎉 所有 API 测试完毕，机器人在待命状态。")

if __name__ == '__main__':
    main()
