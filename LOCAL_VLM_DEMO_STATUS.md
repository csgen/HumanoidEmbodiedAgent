# 本地 VLM 视频到机器人响应 Demo 状态说明

## 当前目标

本分支当前已经打通如下演示链路：

`输入视频 -> 本地 VLM 直接生成机器人控制代码 -> Webots 中 NAO 执行 -> 桌面双窗口录屏`

这个 Demo 的目标不是“机器人模仿人类动作”，而是“机器人像宠物一样，对人类动作做出合适回应”。

## 已完成内容

- 新增本地 VLM 响应链路，支持直接从多帧视频生成机器人控制代码。
- 新增 one-shot 视频采样流程，支持直接从本地视频文件均匀采样帧送入 VLM。
- 新增桌面对照演示脚本：左侧播放测试视频，右侧运行 Webots / NAO。
- 新增批量录屏脚本，能够自动生成多条 mp4 演示结果。
- 整理了 10 条简单的人类动作测试视频，便于快速调试响应逻辑。

## 关键目录

- 输入示例视频：`debug_video_samples/`
- 用户自拍视频样例：`example_video/`
- 双窗口录屏结果：`artifacts/screen_recordings/`
- 本地 VLM / 控制器代码：`nao_VLM/controllers/nao_vlm_test/`
- 演示脚本：`scripts/`

## 这次提交包含的主要脚本

- `scripts/launch_side_by_side_demo.sh`
  - 启动左侧视频 + 右侧 Webots 对照演示。
- `scripts/batch_record_side_by_side_demos.sh`
  - 批量遍历 `debug_video_samples/*.mp4` 并录屏。
- `scripts/screen_record_region.py`
  - 按区域录制桌面，输出 mp4。
- `scripts/local_vlm_codegen_server.py`
  - 本地 HTTP 包装，便于后续把本地 VLM 作为服务使用。

## 当前观察到的问题

虽然链路已经跑通，但机器人行为现在还不够自然，主要问题有两类：

1. **动作模板过于夸张**
   - 部分动作使用了较大的肩部摆幅。
   - 部分动作调用了手爪开合、下肢弯曲、前后平移，视觉上会显得突兀。
   - 个别响应更像“机械演示动作”，不像自然的社交反馈。

2. **本地 VLM 直出控制代码的稳定性还不够**
   - 当前本地小模型对时序理解有限。
   - 对简单动作可以生成可执行结果，但复杂输入上仍可能输出不稳定或不自然的控制序列。

## 本轮调试结论

这次进一步确认，机器人行为看起来奇怪，根因不只一个：

1. **中间标签层不符合最终目标**
   - 用户要求的理论目标是：`视频 -> VLM -> 机器人关节 / 原语控制代码`。
   - 中间再做人类手势标签，会把系统变成半硬编码，不是纯 VLM 控制。
   - 因此当前已经回退到“本地 VLM 直接生成 JSON + Python 控制程序”的方向。

2. **旧动作模板过于戏剧化**
   - 旧版本里有手爪开合、明显下肢动作、前后平移、较大的肩部摆动。
   - 这些动作虽然“显眼”，但不自然，像在做夸张演示而不是社交回应。

## 本轮已完成的修复

### 1) 取消中间标签层

当前代码方向已经调整为：

`视频帧 -> 本地 VLM 直接输出 semantic JSON + Python 控制代码`

这更符合最终目标：

- 不预设固定手势类别
- 不把机器人响应硬编码成几个模板动作
- 允许 VLM 针对任意输入视频生成关节级控制程序

### 2) 保留低层原语接口

虽然不再做中间标签，但系统仍然保留低层控制原语：

- `move_joint`
- `move_joints`
- `move_arm_ik`
- `oscillate_joint`
- `hold`
- `idle`
- `speak`
- `navigate_to`

这里的关键点是：

- 这些原语只是执行接口，不是高层硬编码动作类别。
- VLM 需要直接决定用哪些关节、以什么角度、什么时长来组合行为。

## 当前离线验证结果

当前更关键的验证目标是：

- 本地 VLM 是否能稳定输出可解析的 `json + python`
- 输出的 Python 是否只使用允许的低层原语
- 执行动作是否在 Webots 中自然且不越界

## 仍然待继续优化的问题

目前仍待继续优化的重点是：

- 本地小模型生成结构化控制代码的成功率
- 多帧时序理解质量
- 生成代码的自然度与可执行性

## 目前最适合展示的样例

以下几条最适合继续做直出控制调试：

- `debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4`
- `debug_video_samples/pointing__pointing_gesture__emA8oMXjnb4.mp4`
- `debug_video_samples/stop__stop_palm_gesture__j7QHtHhw5as.mp4`
- `debug_video_samples/thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4`

对应已有录屏结果位于：

- `artifacts/screen_recordings/`

## 后续调试方向

下一步优先做两件事：

1. 提高本地 VLM 直出 `json + python` 的成功率。
2. 优先稳定 4 个代表性样例，再扩展到其余样例。

## 运行方式

单条对照演示：

```bash
bash scripts/launch_side_by_side_demo.sh debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4
```

批量录屏：

```bash
bash scripts/batch_record_side_by_side_demos.sh
```

## 分支说明

当前所有相关改动均保留在：

- `br_dev_setup_darian`
