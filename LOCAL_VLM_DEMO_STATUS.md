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
