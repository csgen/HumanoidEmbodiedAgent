import pinocchio as pin
import numpy as np
import os

def main():
    # 指向官方 nao_robot 仓库
    package_dir = os.path.expanduser("~/nao_VLM/nao_robot")
    # NAO 官方模型通常放在这个路径下
    urdf_path = os.path.join(package_dir, "nao_description/urdf/naoV50_generated_urdf/nao.urdf")
    
    # 做个双保险：如果上一个路径没有，尝试根目录下的 nao.urdf
    if not os.path.exists(urdf_path):
        urdf_path = os.path.join(package_dir, "nao_description/urdf/nao.urdf")
        if not os.path.exists(urdf_path):
            print(f"找不到 URDF 文件！请手动检查 nao_description/urdf/ 目录")
            return

    print(f">>> 正在加载本地标准 NAO 模型: {urdf_path}")
    
    # 加载模型 (声明这是一个带有 6 DoF 浮动基座的机器人)
    model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
    data = model.createData()

    print(f"成功加载 NAO 模型！总自由度 (nv): {model.nv}")

    # 创建默认安全姿态
    q_neutral = pin.neutral(model)
    
    # 计算质心
    pin.centerOfMass(model, data, q_neutral)
    
    com_position = data.com[0]
    print(f"当前默认姿态下的质心坐标 (X, Y, Z):")
    print(np.round(com_position, 4))

    print(f"\nNAO 的前 5 个关节名称示例: {model.names[1:6]}")

if __name__ == '__main__':
    main()
