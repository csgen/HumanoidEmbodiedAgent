# 本地 VLM 视频到机器人响应 Demo 状态说明

## 当前目标

本分支当前已经打通如下演示链路：

`输入视频 -> 本地 VLM 直接生成机器人控制代码 -> Webots 中 NAO 执行 -> 桌面双窗口录屏`

这个 Demo 的目标不是“机器人模仿人类动作”，而是“机器人像宠物一样，对人类动作做出合适回应”。

## 当前所处 Phase

对照 `Humanoid Embodied Agent — Proposal vs Current Implementation Analysis & Refined Plan.md`，
当前工作最准确地属于：

- `Phase 3` 收尾：prompt redesign + sandbox hookup
- `Phase 4` 进行中：fallback + safety

还没有进入完整的 `Phase 5` 评测框架，也没有做完 `Phase 6` stretch goals。

## 阶段清单

- [x] Phase 0：模块整合 + 视觉输入
- [x] Phase 1：运动原语
- [x] Phase 2：idle + 多线程编排
- [x] Phase 3：prompt redesign + sandbox hookup
- [ ] Phase 4：fallback + safety（当前进行中）
- [ ] Phase 5：评测框架（预录视频）
- [ ] Phase 6：stretch goals

当前重点仍然是“模型质量和演示打磨”，还不是完整的 Phase 5 收尾。

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
  - 现在默认启用 shortlist，只优先录制当前更稳的 3 条演示样例。
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

### 3) 本地 VLM 多候选 + VLM 自选优

这轮新增了几项专门面向“不要所有视频都退化成同一段腕部动作”的改进：

- 本地 VLM 从单候选改为多候选生成。
- 对多个候选，先做安全/结构合法性过滤。
- 再把候选列表重新交给 VLM，让 VLM 自己从候选里选出最贴合视频、最自然的一条。
- 给每段视频提取低层时序摘要：
  - `motion_energy`
  - `dominant_axis`
  - `lateral_bias`
  - `vertical_bias`
  - `activity_level`
- 把这些低层摘要作为弱约束写回 prompt，但不再额外注入“动作先验解释文字”，避免把低层摘要偷偷变成半硬编码动作规则。

这里仍然没有引入“手势类别 -> 固定机器人动作模板”的硬编码映射。
当前链路更接近：

`视频 -> VLM 生成多个控制程序 -> VLM 自选优 -> sandbox 校验 -> 执行`

### 4) 静态校验与修复增强

- 新增 sandbox 预检：
  - 禁止 `navigate_to`
  - 禁止下肢关节
  - 检查 joint limits
  - 检查异常 oscillation 振幅 / 频率 / 时长
  - 检查无意义 IK 目标
- 新增对 keyword arguments 的静态检查，避免 `move_arm_ik(side='left', xyz=[0,0,0], ...)` 这类无效代码误通过。
- 新增本地 VLM 的 repair / refinement 路径：
  - 首次代码不可执行时，仍然交给 VLM 自修复
  - 候选过于 generic 时，再让 VLM 基于同一视频二次重写
- 新增“没有任何有效候选时也触发 repair”逻辑，避免直接失败退出。
- 新增“修复 / refinement 多候选择优”逻辑，避免一次修复失败后直接接受较差结果。
- 新增 joint alias 归一化，减少小模型输出关节名不标准导致的失败。
- 新增常见伪代码归一化：
  - `move_head(...)` 会被转成合法的 `move_joints({'HeadYaw': ..., 'HeadPitch': ...}, ...)`
  - 被错误生成为 Python 集合字面量的一组 primitive 调用，会被提取成逐行可执行代码

### 5) 机器人动作发怪的一个执行层根因已修

本轮确认了一个会直接导致“看起来很怪”的执行层问题：

- 旧版 `move_arm_ik(...)` 在求解时可能带动不相关关节。
- 这会让 VLM 明明只想抬一只手，结果却把更多上半身关节一起拖动，观感会很机械。

本轮已修成：

- `move_arm_ik('left', ...)` 只允许左臂链参与解算
- `move_arm_ik('right', ...)` 只允许右臂链参与解算
- 最终只把目标侧手臂关节写回 Webots 电机

这不会 magically 让小模型变聪明，但会明显减少“同一条 VLM 代码执行出来却像乱动”的情况。

### 6) 这轮进一步去掉了“不是当前视频也能触发旧动作”的风险

本轮又继续做了三件重要事情：

- 运行时 sandbox 不再暴露旧兼容接口：
  - `look_at`
  - `move_arm`
  - `operate_gripper`
  - `set_posture`
  - `speak`
  - `navigate_to`
- fallback 不再 replay 上一次成功代码。
- 新增低层手部原语 `set_hand(side, openness, duration)`，让 VLM 仍然只通过低层 primitive 控制手部，而不是走旧高层抓手接口。

这三点的意义是：

- 当前动作更严格地绑定在“本次视频 -> 本次 VLM 代码 -> 本次执行”这条链路上。
- 失败时不会偷偷放上一次旧动作，看起来像机器人“有反应”，实际上却不是当前视频控制。
- 手部动作仍可表达，但表达入口已经是低层原语，不是硬编码行为函数。

### 7) 新增一层代码净化器，专门把小模型的低级错误修成可执行 primitive 代码

由于本地 3B 小模型常常会生成一些低级但可修复的问题，本轮新增了 AST 级代码净化器，仍然不引入手势模板，只做低层合法化修正：

- 关节名别名归一化：例如把 `head` 修成 `HeadYaw`
- `move_arm_ik` 左右手目标的 `y` 符号修正
- `move_arm_ik` 的 `xyz` 范围裁剪到更合理的上半身空间
- `oscillate_joint` 的 center / amplitude 自动裁剪，避免腕关节中心值直接贴边越界
- `set_hand` openness 自动裁剪到 `[0, 1]`

注意：

- 这层不是“语义模板映射”，不是把 waving/pointing/stop 映射到固定动作。
- 它只是把 VLM 已经写出来的低层 primitive 代码做合法化和落地修正。
- 因此它仍然符合“视频 -> VLM -> primitive 代码 -> 执行”的主原则。

### 8) 新增“自然性筛选”但仍不引入语义动作模板

为了进一步逼近“像宠物一样自然”的目标，本轮又新增了一层候选自然性筛选：

- 它不看手势标签，不做 `wave -> 某固定动作` 这种映射。
- 它只看 VLM 已生成的 primitive 代码模式本身，例如：
  - 是否只有单关节振荡
  - 是否只有 `hold(...)`
  - 是否没有 arm/head 协同
  - 是否调用过多、过碎、过机械
- 这个评分只用于：
  - 有多个合法候选时，优先保留更像“短、平滑、上半身协同回应”的那条
  - 当结果过于机械时，触发 refinement，让 VLM 基于同一视频重写

这层的作用类似“低层运动学审查器”，而不是语义行为模板库。

### 9) 本机现在已经可以跑更强的本地 VLM：`Qwen2.5-VL-7B-Instruct` + 4-bit

本轮一个最重要的实质性进展是：

- 已确认本机 `RTX 4060 Laptop 8GB` 可以加载并运行 `Qwen/Qwen2.5-VL-7B-Instruct`
- 方式是 `4-bit quantization`
- 已把默认 demo / debug 配置切到：
  - `LOCAL_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct`
  - `LOCAL_VLM_LOAD_IN_4BIT=1`

### 10) 本轮继续做了“模型质量和演示打磨”

这轮不是继续堆新功能，而是把已经打通的主链路往“更自然、能录屏汇报”这个方向继续收紧：

- 强化了 system / refinement / selection prompt，明确压制以下几类结果：
  - 连续重复完全相同的 primitive 调用
  - 没有画面依据的双臂镜像动作
  - 只有肘/腕小摆动、缺乏可读姿态的 generic 代码
- 新增了代码级自然性检查与质量排序：
  - `_has_consecutive_duplicate_calls(...)`
  - `_has_symmetric_dual_arm_ik(...)`
  - `_has_mirrored_dual_arm_posture(...)`
  - `_uses_only_distal_arm_joints(...)`
- 新增了“舒适区裁剪”，进一步抑制头部、肩部、肘部的极限角度。
- refinement / repair 不再“只要合法就覆盖原结果”，而是必须在质量评分上更好，才允许替换原候选。
- 当第一次 refinement 仍然 generic 时，会自动触发一次带明确质量反馈的 second-pass refinement。
- refinement / repair 现在会对同一个 prompt 进行多次 VLM 采样，再从多个生成结果里挑较优者，而不是单发接受。

### 11) Phase 4 真实推进：retry 已接上主循环

这轮不只是继续改 prompt，也补上了一个真实的 phase 4 缺口：

- 之前 `FallbackPolicy.handle_failure(...)` 虽然会返回 `retry`，但主循环并没有真正把同一批 frames 重新送回 worker。
- 现在已经补成：
  - 触发时缓存最近一次真正送入 VLM 的 frame 序列
  - `call_failed` 或 `exec_failed` 时，如果 fallback 决策是 `retry`
  - 就会用同一批 frames 重新 `kick_with_frames(...)`

这更符合 phase list 里“Tier A retry on same evidence”的要求，且仍然不引入任何手势模板或硬编码动作映射。

这一步的意义是：

- 仍然没有回退到“手势标签 -> 固定动作模板”。
- 但会更严格地逼 VLM 生成“可读、自然、可演示”的上半身 primitive 程序。

## 本轮离线回归结论

基于 `Qwen/Qwen2.5-VL-7B-Instruct` + `4-bit`，当前样例大致分成三类：

### A. 适合优先录屏演示的样例

1. `debug_video_samples/thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4`
   - 当前可得到较短、较克制的右臂响应
   - 虽然还不算完美，但已经具备演示价值

2. `debug_video_samples/finger_no__no_no_finger_wave__82vLYYXukIE.mp4`
   - 当前可得到较清晰的单侧上半身回应
   - 比 `waving` 更稳定，更适合先拿来录屏汇报

3. `debug_video_samples/stop__stop_palm_gesture__j7QHtHhw5as.mp4`
   - 当前输出能跑通，且画面上有可见响应
   - 但仍需继续压大幅度关节极限动作

4. `debug_video_samples/beckon__woman_come_here_beckon__9CeeTCQskFs.mp4`
   - 当前能生成可执行、多 primitive 的上半身回应
   - 但语义理解还有偏差，适合作为备选，不建议放在第一顺位

### B. 已跑通，但还不推荐优先演示的样例

1. `example_video/webcam_20260425_072825.mp4`
   - 仍然偏 generic
   - 当前更像“简单肘部收放”，不够像自然社交回应

2. `debug_video_samples/pointing__pointing_gesture__emA8oMXjnb4.mp4`
   - 比最早期好一些，但依旧容易退化成 generic 的双侧上肢姿态
   - 后续应继续打磨为“单侧注意 / 指向回应”而不是镜像姿态

3. `debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4`
   - 目前仍会周期性退化成双侧 generic 上半身姿态
   - 这条样例暂时不应作为主演示数据，除非后续回归再次稳定

### C. 当前不建议拿来录屏汇报的样例

- `debug_video_samples/no_shake__man_shake_head_no__yZ-351AUZqE.mp4`
- `debug_video_samples/yes_nod__woman_nod_yes__Ouk-bdR3L30.mp4`

这些样例目前对本地 7B 仍然容易产生过于 generic 的响应，不适合当前阶段当作主演示数据。

## 下一步打磨重点

下一步最该做的，不是回退到模板，而是继续做两件事：

1. **上半身观感约束继续收紧**
   - 进一步抑制肩关节极限值
   - 减少“单帧看着显眼、连起来很怪”的姿态
   - 更鼓励 head + dominant arm 的 companion-style 响应

2. **只挑最适合汇报的样例录屏**
   - 当前更稳、更适合优先录屏的是：
     - `debug_video_samples/thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4`
     - `debug_video_samples/finger_no__no_no_finger_wave__82vLYYXukIE.mp4`
     - `debug_video_samples/stop__stop_palm_gesture__j7QHtHhw5as.mp4`
   - 备选可尝试加入：
     - `debug_video_samples/beckon__woman_come_here_beckon__9CeeTCQskFs.mp4`
   - 这样能先确保你对队友展示的是“已有可见进展”，而不是把还不稳定的样例一并暴露出来。

这件事很关键，因为到这里为止，系统的“自然度”瓶颈已经不再只是代码逻辑，而更大程度上是模型本身的多帧理解能力。
相比继续叠更多规则，直接换更强的 VLM 更符合你的要求：

- 仍然保持 `视频 -> VLM -> primitive 代码 -> 执行`
- 不引入手势标签模板
- 不回退到硬编码动作库

### 10) 7B 回归结果

这轮用 `Qwen2.5-VL-7B-Instruct` 做了真实回归：

- `example_video/webcam_20260425_072825.mp4`
  - 已跑通
  - 输出不再是单纯 wrist 越界振荡
  - 当前得到的是更保守的上半身响应：双肘轻微调整 + `idle(...)`
- `debug_video_samples/pointing__pointing_gesture__emA8oMXjnb4.mp4`
  - 已跑通
  - 当前得到的是更短、更克制的单臂指向式上半身响应：`move_arm_ik('right', ...) + idle(...)`

虽然还不能说“已经像成熟宠物机器人一样自然”，但这已经比 3B 时期更接近“短、克制、非夸张、可展示”的方向。

## 当前离线验证结果

当前更关键的验证目标是：

- 本地 VLM 是否能稳定输出可解析的 `json + python`
- 输出的 Python 是否只使用允许的低层原语
- 执行动作是否在 Webots 中自然且不越界

本轮离线回归重点测试了 3 条代表视频 + 1 条用户示例视频：

- `debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4`
- `debug_video_samples/pointing__pointing_gesture__emA8oMXjnb4.mp4`
- `debug_video_samples/stop__stop_palm_gesture__j7QHtHhw5as.mp4`
- `example_video/webcam_20260425_072825.mp4`

当前结果：

- `waving`
  - 当前已经能生成 `move_joint + set_hand + oscillate_joint` 这种纯低层原语组合。
- `pointing`
  - 候选不再允许无效 `IK` 目标混进成功结果。
  - 当前已经能通过净化器把 `head` 这类小模型错误别名修成合法关节名，并输出可执行结果。
- `stop`
  - 当前已能生成与 waving / pointing 不同的抑制性上半身动作。
  - 在这轮回归里，最稳妥的有效结果仍偏保守，常见为短暂 `hold(...)`。
- 多候选选择
  - 现在已经不是“只要能执行就选”，而是会在多个可执行候选中优先选更自然的 primitive 组合。
- `example_video`
  - 主链路已能挡住之前那种“集合字面量伪代码被当成成功”的问题。
  - 由于 3B 本地模型较弱，当前输出仍偏简单，且仍有失败样例，但链路约束更可靠了。

这说明当前分支相比主分支，已经出现了可展示的新进展：

- 不再只是单一路径的腕部摆动
- VLM 生成代码开始对不同输入视频表现出差异化
- 差异化仍然通过 VLM 直出控制代码完成，而不是手势标签模板
- 候选选优也交给 VLM 本身，而不是手写动作分类器或模板映射

## 本轮最新 checkpoint

本轮继续推进后，当前最准确的阶段判断仍然是：

- `Phase 3` 收尾
- `Phase 4` 推进中

还没有进入完整的 `Phase 5` 评测框架，因此这轮工作仍然属于“控制链路稳定性 + 安全/回退 + 模型质量打磨”，而不是完整 benchmark 收尾。

这轮新增的实质性进展有：

- 修复了 `move_arm_ik(side='L'/'R', ...)`、`set_hand(side='L'/'R', ...)` 这类小模型常见 side 输出导致的失败。
  - 现在 sanitize 层、sandbox validate 层、运行执行层都能把 `L/R/left_arm/right_hand` 统一归一化到 `left/right`。
- 进一步强化了“非模板式”的候选质量筛选。
  - 新增对“双臂碎片化动作”的惩罚。
  - 新增对“laterality 明显但代码主侧不清晰”的惩罚。
  - 新增对“非主侧手臂无依据乱配合”的惩罚。
  - 新增对“只有 hand open/close 但没有肩/头姿态锚点”的惩罚。
- 继续新增了“运行时签名一致性”校验，并已并入主链路。
  - 不再只依赖宽松的静态 validator。
  - 现在主链路和离线脚本都会检查生成代码是否真的匹配 primitive 接口签名。
  - 这修掉了“静态通过，但真实执行参数不匹配”的一类假阳性。
- 重新离线回归了更多样例，并重新生成了 6 条桌面对照录屏：
  - `artifacts/screen_recordings/thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4`
  - `artifacts/screen_recordings/finger_no__no_no_finger_wave__82vLYYXukIE.mp4`
  - `artifacts/screen_recordings/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4`
  - `artifacts/screen_recordings/pointing__pointing_gesture__emA8oMXjnb4.mp4`
  - `artifacts/screen_recordings/no_shake__man_shake_head_no__yZ-351AUZqE.mp4`
  - `artifacts/screen_recordings/yes_nod__woman_nod_yes__Ouk-bdR3L30.mp4`

## 本轮核心离线回归结果

### 1) `finger_no__no_no_finger_wave__82vLYYXukIE.mp4`

- 之前会因为 `invalid_arm_side: L` 直接失败。
- 当前已恢复为可执行、单侧更清晰的响应：

```python
move_arm_ik('left', [0.06, 0.06, 0.13], 1.0)
hold(0.5)
move_arm_ik('left', [0.06, 0.06, 0.0], 1.0)
```

- `validator_ok=True`
- `generic_like=False`
- 当前已经比之前“双臂一起乱配”的结果更适合演示。

### 2) `stop__stop_palm_gesture__j7QHtHhw5as.mp4`

- 当前能收敛到更短、更可读的单侧 stop-like 响应：

```python
move_arm_ik('right', [0.1, -0.06, 0.1], 0.5)
hold(0.5)
```

- `validator_ok=True`
- `generic_like=False`
- 相比之前带额外左臂配合姿态的版本，当前更克制，也更像“看到 stop 手势后的宠物式停住/回应”。

### 3) `thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4`

- 当前在继续打磨后，已经不再停留在“只有肘腕/手爪动作”的状态。
- 一次最新回归得到的是：

```python
move_arm_ik('left', [0.06, 0.18, 0.1], 0.5, 'cubic')
set_hand('left', 0.5, 0.5, 'cubic')
idle(0.5)
```

- `validator_ok=True`
- `generic_like=False`
- 这仍然不是完美答案，但已经比纯 distal-joint 反应更可读。

### 4) `waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4`

- 这条之前经常退化成 generic 双侧姿态或碎片化动作。
- 当前已经能稳定收敛到更简洁的单侧抬手响应，例如：

```python
move_joints({'LShoulderPitch': -0.5, 'LElbowRoll': -0.5, 'LWristYaw': 0.5, 'HeadPitch': 0.2}, 0.5, 'cubic')
idle(0.5)
```

- 这条还不是最强主演示，但已经比之前明显自然。

### 5) `pointing__pointing_gesture__emA8oMXjnb4.mp4`

- 当前能够输出更明确的单侧指向式上半身动作，例如：

```python
move_arm_ik('right', [0.06, -0.06, 0.0], 1.5)
move_joints({'RElbowYaw': 1.2, 'RShoulderPitch': -1.0, 'RShoulderRoll': -0.5}, 1.5)
hold(0.5)
idle(1.0)
```

- 当前可执行且可演示，但观感仍有继续细化空间。

### 6) `no_shake__man_shake_head_no__yZ-351AUZqE.mp4` / `yes_nod__woman_nod_yes__Ouk-bdR3L30.mp4`

- 这两条低运动量样例现在已经更偏“头部主导、安静回应”，不再乱用手臂：

```python
move_joints({'HeadYaw': 0.0, 'HeadPitch': -0.1}, 0.1, 'min_jerk')
hold(0.2)
```

- 这更符合“宠物式轻微关注/点头感”的演示方向。

## 仍然待继续优化的问题

目前仍待继续优化的重点是：

- 本地小模型生成结构化控制代码的成功率
- 多帧时序理解质量
- 生成代码的自然度与可执行性
- `thumbs_up` 这类样例虽然已有明显改进，但仍可能在不同采样次之间回落
- `stop` / `example_video` 这类样例虽然已跑通，但当前动作仍有不稳定轮次
- 当前离线改进已明显优于之前，但还不能宣称“任意视频都能自然稳定反应”

## 本轮收尾状态

到当前 checkpoint 为止：

- “机器人动作 fully controlled by VLM, not hardcoded” 这一点已经继续被压实。
  - 没有重新引入 gesture label / action template / rule-based dispatch。
  - 反而进一步把签名校验、候选择优和自然性约束做到了 VLM 直出 primitive 代码这条链路内部。
- “机器人只做上半身响应、不走路” 这一点已经稳定满足。
  - 当前所有筛选、校验、演示脚本都围绕头、手臂、手、上半身姿态。
- “像宠物一样自然” 这点有明显进步，但仍是当前 Phase 4 内最主要的剩余难题。
  - 现在已经能稳定产出一批可展示的上半身反应。
  - 但还不能宣称每条输入都达到了成熟产品级自然性。

换句话说：

- “不是硬编码模板” 这一点已经明确达成
- “不同输入能明显产生不同响应” 这一点已经有进展，而且这些响应现在更严格地来自本次 VLM 代码，而不是历史 replay 或旧接口捷径
- “所有样例都像成熟宠物机器人一样自然” 这一点还没有达成，主要瓶颈已经更明确地收敛到本地 7B 视频模型的时序理解与代码生成质量，而不是控制链路还在偷偷硬编码

## 目前最适合展示的样例

以下几条当前更适合展示或继续做直出控制调试：

- `debug_video_samples/thumbs_up__happy_man_thumbs_up__W09XgqL0cxg.mp4`
- `debug_video_samples/finger_no__no_no_finger_wave__82vLYYXukIE.mp4`
- `debug_video_samples/waving__portrait_guy_waving_hand__QvJaZ0h94Eo.mp4`
- `debug_video_samples/pointing__pointing_gesture__emA8oMXjnb4.mp4`
- `debug_video_samples/no_shake__man_shake_head_no__yZ-351AUZqE.mp4`
- `debug_video_samples/yes_nod__woman_nod_yes__Ouk-bdR3L30.mp4`

以下样例暂时不建议当主演示：

- `debug_video_samples/stop__stop_palm_gesture__j7QHtHhw5as.mp4`
- `example_video/webcam_20260425_072825.mp4`

对应已有录屏结果位于：

- `artifacts/screen_recordings/`

## 后续调试方向

下一步优先做两件事：

1. 优先继续稳定 `thumbs_up / yes_nod / no_shake` 这类细粒度样例。
2. 继续把“单侧主导 + 另一侧安静 + 头部轻微辅助”这个非模板式自然性约束做得更稳。
3. 在 Webots 中复核当前离线最佳候选的观感，继续修“动作看着怪”的问题。

如果后续本地 3B 模型仍然明显不够，可以考虑：

- 保持“VLM 直出控制代码”这个原则不变
- 但更换成更强的本地视频模型或量化模型
- 仍然不回退到“手势标签 -> 固定动作模板”方案

## 新增调试脚本

- `scripts/offline_local_vlm_debug.py`
  - 直接把本地 mp4 送入本地 VLM
  - 打印 `semantic_context`
  - 打印生成的 Python 控制代码
  - 用静态 validator 检查代码是否可接受
  - 用严格签名 runtime executor 再检查一遍，避免“静态过、运行炸”

适合在 conda 环境中做快速离线回归。

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
