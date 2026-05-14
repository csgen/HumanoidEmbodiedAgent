# 协作者使用指南

## English Summary

This guide is the day-to-day collaborator runbook. The current work is Phase 5:
adding a reproducible evaluation framework around the existing Webots + NAO +
VLM controller. The important entry points are `nao_VLM/controllers/nao_vlm_test/`
for runtime control, `evaluation/` for benchmark/metrics/judge code, and
`scripts/` for demos and recording. Paths are now repo-relative by default; set
`REPO_DIR`, `WEBOTS_BIN`, or `PYTHON_BIN` only when your machine needs an
override.

## 这项目做什么

输入视频 → VLM → NAO 控制代码 → Webots 执行 → 左右分屏录屏。

目标是让机器人像宠物一样自然回应，不是硬编码动作模板。

## 当前分支

- 开发分支：`br_dev_setup_darian`
- 主分支：`main`
- 当前阶段：Phase 5 评测框架进行中

## 项目结构

- `nao_VLM/controllers/nao_vlm_test/`：Webots 控制器和 VLM 执行链路
- `scripts/`：录屏、预计算、启动演示等脚本
- `example_video/`：单个示例视频
- `debug_video_samples/`：调试样例视频
- `artifacts/screen_recordings_matched/`：最终对外展示的录屏成片
- `evaluation/`：Phase 5 benchmark、metrics、rule baseline、VLM-as-Judge
- `LOCAL_VLM_DEMO_STATUS.md`：当前阶段和近期进展
- `Humanoid Embodied Agent — Proposal vs Current Implementation Analysis & Refined Plan.md`：总阶段规划

## 环境

1. 创建环境：`conda env create -f environment.yml`
2. 激活环境：`conda activate humanoid_robot_vlm_darian`
3. 确认 `.env`、`nao_VLM/controllers/nao_vlm_test/runtime.ini` 已指向正确 Python
4. 确认 Webots 可执行文件可通过 `webots` 找到，或设置 `WEBOTS_BIN=/path/to/webots`

## 常用入口

- 录制双窗口演示：`bash scripts/record_matched_oneshot_demo.sh <输入视频> <输出 mp4>`
- 预计算后录制：`bash scripts/record_precomputed_side_by_side_demo.sh <输入视频> <输出 mp4>`
- 仅启动演示窗口：`bash scripts/launch_side_by_side_demo.sh <输入视频>`
- 录摄像头样例：`python3 scripts/record_webcam.py`
- 批量生成对照录屏：`bash scripts/batch_record_side_by_side_demos.sh`
- 直播摄像头 demo：`WEBCAM_SOURCE=0 bash scripts/run_live_camera_demo.sh`
- Phase 5 CaP 评测：`python -m evaluation.run_benchmark --scenario-set pilot --rounds 1 --method cap`
- Phase 5 baseline：`python -m evaluation.run_benchmark --scenario-set pilot --rounds 1 --method rule_baseline`
- Judge 报告：`python -m evaluation.judge artifacts/eval/*.json --output artifacts/eval/report.md`

## 路径与环境变量

脚本现在默认从自身位置推导项目根目录。只有在特殊环境中才需要手动设置：

- `REPO_DIR=/path/to/HumanoidEmbodiedAgent`
- `WEBOTS_BIN=/path/to/webots`
- `PYTHON_BIN=/path/to/python`
- `WEBCAM_SOURCE=0` 本机 Linux 摄像头
- `WEBCAM_SOURCE=http://127.0.0.1:5000/video_feed` SSH tunnel 后的远程摄像头

## 录屏产物

- 只推送：`artifacts/screen_recordings_matched/`
- 其他 `artifacts/` 目录默认视为本地调试产物，不随便提交

## 协作约定

- 先 `git pull origin main` 再改代码
- 代码改动留在 `br_dev_setup_darian`
- 合并到 `main` 前先确认录屏产物和文档都已更新
- 不要把 `artifacts/screen_recordings/`、`artifacts/oneshot/` 之类的临时调试结果一起推上去
- 录屏只保留 `artifacts/screen_recordings_matched/`

## 协作流程

1. `git pull origin main`
2. 在 `br_dev_setup_darian` 上改代码
3. 只提交必要代码、文档和 `artifacts/screen_recordings_matched/`
4. `git push origin br_dev_setup_darian`
5. 需要合并到主分支时，再把 `br_dev_setup_darian` 合入 `main`
