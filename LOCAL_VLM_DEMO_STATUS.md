# 本地 VLM 视频到机器人响应 Demo 状态说明

## 当前目标

本分支当前已经打通如下演示链路：

`输入视频 -> 本地 VLM 判别人的动作意图 -> 生成机器人响应动作序列 -> Webots 中 NAO 执行 -> 桌面双窗口录屏`

这个 Demo 的目标不是“机器人模仿人类动作”，而是“机器人像宠物一样，对人类动作做出合适回应”。

## 已完成内容

- 新增本地 VLM 响应链路，支持直接从多帧视频判断人的动作意图。
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
- `scripts/local_vlm_label_server.py`
  - 本地 HTTP 包装，便于后续把本地 VLM 作为服务使用。

## 当前观察到的问题

虽然链路已经跑通，但机器人行为现在还不够自然，主要问题有两类：

1. **动作模板过于夸张**
   - 部分动作使用了较大的肩部摆幅。
   - 部分动作调用了手爪开合、下肢弯曲、前后平移，视觉上会显得突兀。
   - 个别响应更像“机械演示动作”，不像自然的社交反馈。

2. **局部分类还不稳定**
   - `waving / pointing / stop / thumbs_up` 基本可用。
   - `no_shake / shrug / yes_nod / finger_no / beckon` 仍有误判现象。
   - 因为标签错了，即使执行动作本身没问题，也会表现得“不对题”。

## 本轮调试结论

这次进一步确认，机器人行为看起来奇怪，根因不只一个：

1. **之前的分类目标设计不合理**
   - 旧版本让本地 VLM 直接输出“机器人应该采取的响应标签”。
   - 这会把“看见了什么手势”和“机器人该怎么回应”混在同一步里。
   - 一旦模型理解偏了，就会直接跳到错误的机器人动作上。

2. **旧动作模板过于戏剧化**
   - 旧版本里有手爪开合、明显下肢动作、前后平移、较大的肩部摆动。
   - 这些动作虽然“显眼”，但不自然，像在做夸张演示而不是社交回应。

## 本轮已完成的修复

### 1) 把识别与决策拆开

新的本地流程改成：

`视频帧 -> 识别人类手势标签 -> 映射到机器人响应标签 -> 生成动作模板`

当前人类手势标签包括：

- `HUMAN_WAVE_HELLO`
- `HUMAN_POINT_DIRECTION`
- `HUMAN_BECKON_COME`
- `HUMAN_STOP_PALM`
- `HUMAN_REJECT_NO`
- `HUMAN_POSITIVE_ACK`
- `HUMAN_SHRUG_UNCERTAIN`
- `HUMAN_UNCLEAR`

这样做的好处是：

- 更容易调试到底是“看错了”还是“回应动作不好看”。
- 更容易逐项修正分类规则。
- 更适合以后继续替换更强的 VLM 或时序模型。

### 2) 收敛机器人动作模板

本轮已经明显削弱/移除了以下容易显怪的内容：

- 大幅下肢动作
- 真实前后平移
- 手爪开合表演
- 过大的肩部摆幅

现在动作更偏向：

- 轻微抬臂
- 小幅挥手
- 轻微点头
- 头部朝向与停顿
- 小幅困惑歪头

## 当前离线验证结果

以下 4 条代表样例已经离线验证通过，分类与响应映射合理：

- `waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4`
  - `HUMAN_WAVE_HELLO -> PET_GREET_HAPPY`
- `pointing__pointing_gesture__emA8oMXjnb4.mp4`
  - `HUMAN_POINT_DIRECTION -> PET_ORIENT_FOLLOW`
- `stop__stop_palm_gesture__j7QHtHhw5as.mp4`
  - `HUMAN_STOP_PALM -> PET_FREEZE_RESPECTFUL`
- `thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4`
  - `HUMAN_POSITIVE_ACK -> PET_EXCITED_ACK`

另外两条也比之前更合理：

- `shrug__shrugging_person__bx_US6Mwdhk.mp4`
  - 现在识别为 `HUMAN_SHRUG_UNCERTAIN`
- `yes_nod__woman_nod_yes__Ouk-bdR3L30.mp4`
  - 现在识别为 `HUMAN_POSITIVE_ACK`

## 仍然待继续优化的问题

以下几条仍不够稳定，适合后续继续做：

- `beckon__woman_come_here_beckon__9CeeTCQskFs.mp4`
- `finger_no__no_no_finger_wave__82vLYYXukIE.mp4`
- `no_shake__man_shake_head_no__yZ-351AUZqE.mp4`

这些问题更像是“细粒度时序识别”问题，而不是动作模板问题。

## 目前最适合展示的样例

以下几条最适合用来继续调自然度：

- `debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4`
- `debug_video_samples/pointing__pointing_gesture__emA8oMXjnb4.mp4`
- `debug_video_samples/stop__stop_palm_gesture__j7QHtHhw5as.mp4`
- `debug_video_samples/thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4`

对应已有录屏结果位于：

- `artifacts/screen_recordings/`

## 后续调试方向

下一步优先做两件事：

1. 把本地动作模板收敛到“轻微、清晰、自然”的宠物式响应。
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
