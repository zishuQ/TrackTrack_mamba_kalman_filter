# Codabench 提交流程（由工作目录内的成功记录还原）

日期：2026-11-17。依据：`outputs/agentguard/experiments/mot20_loo_only_v2_sqrt_size_seed42_bs1024_shard4x1_100e/eval_logs/` 下的完整提交记录（2026-09-14），以及 `outputs/results_*_test/` 中带官方分数的历史 zip。

## 1. 证据与时间线（2026-09-14）

| 时间 | 事件 | 记录文件 |
|---|---|---|
| 15:18 | 评测 watcher 启动，等待训练 | `status.txt` |
| 16:03 | 训练完成，开始 MOT20 all raw 评测 | `status.txt` |
| 16:24 | all raw 评测完成（rc=0） | `status.txt` |
| 16:43 | MOT20 test final+post 完成（rc=0），触发提交 watcher | `status.txt`、`watch_then_submit.sh` |
| 16:43 | 自动 watcher 因文件名检查 bug 退出（`txt names mismatch`，把 `MOT20-04.txt` 与 `MOT20-04` 比较） | `submit_status.txt` |
| 16:46 | 手动提交创建，submission id=925784，status=Submitting | `submit_step1.json`、`submit_step3.json` |
| 16:46–17:04 | 轮询 33 次（间隔 30 s）：Submitting → Running → Finished | `submit_status.txt`、`poll_submission.sh` |
| 17:04 | 保存完整提交记录与官方分数，删除 token | `submit_final.json` |

## 2. 已实测通过的 API 流程

鉴权：所有需要身份的请求带 `Authorization: Token <token>`。token 当时放在 `/tmp/opencode/codabench_token`，用后删除；账号 `jojo0`。

### 步骤 1：预留上传槽位

```bash
curl -sS -X POST https://www.codabench.org/api/datasets/ \
  -H "Authorization: Token <TOKEN>" \
  -F file_size=<zip 字节数> \
  -F name=<提交名称，如 mot20_looonly_v2_sqrt_e100_final_post> \
  -F type=submission \
  -F request_sassy_file_name=<zip 文件名>
```

返回（实测）：

```json
{"key": "121bc81e-...", "sassy_url": "https://miniodis-rproxy.lisn.upsaclay.fr/...&Expires=..."}
```

### 步骤 2：上传 zip 到预签名地址

```bash
curl -sS -o /dev/null -w "%{http_code}" -X PUT "<sassy_url>" \
  -H "Content-Type: application/zip" --data-binary @<zip>
```

实测返回 HTTP 200。

### 步骤 3：创建提交

```bash
curl -sS -X POST https://www.codabench.org/api/submissions/ \
  -H "Authorization: Token <TOKEN>" \
  -F data=<key> \
  -F phase=<PHASE_ID> \
  -F tasks=<TASK_UUID>
```

实测返回：

```json
{"id": 925784, "status": "Submitting", "filename": "...zip", "phase": 16383, "created_when": "2026-09-14 08:45"}
```

已知参数（MOT20）：competition `10050`，phase `16383`（Testing Phase），task `08d6c1f3-6203-4067-824f-f72c30130693`。

### 步骤 4：轮询与收集分数

```bash
curl -sS -H "Authorization: Token <TOKEN>" https://www.codabench.org/api/submissions/<ID>/
```

- 状态值首字母大写：`Submitting` → `Running` → `Finished`；实测评分耗时约 18 分钟。
- 终止判据应为 `status not in {"Submitting", "Running"}`。旧 watcher 用小写 `"finished"` 比较，永远匹配不上（所幸那次是手动轮询脚本完成的）。
- 完成后 JSON 的 `scores` 是列表，每项含 `column_key` / `score` / `is_primary`；分数是字符串、百分比单位（`66.2050000000` = HOTA 66.205）。
- 完整记录还包含 `phase_name`（"Testing Phase"）、`owner`、`is_public`、`created_when`、`filename` 等。

MOT20 id=925784 的官方结果（`submit_final.json`）：HOTA **66.205**、MOTA 77.666、IDF1 82.030、AssA 68.477、DetA 64.189、MT 843、ML 142、CLR_FP 24582、CLR_FN 90305、CLR_Re 82.547、CLR_Pr 94.558、AssRe 72.623、AssPr 82.531、DetRe 69.476、DetPr 79.585、LocA 84.174、IDSW 675、Frag 820。

## 3. 提交包要求（实测）

- zip 内 txt 必须在根目录（无子目录）；
- MOT20 测试 4 个文件：`MOT20-04/06/07/08.txt`；
- MOT17 测试 21 个文件：`MOT17-01/03/06/07/08/12/14` × `DPM/FRCNN/SDP`；
- 每行 10 列（`frame,id,x1,y1,w,h,score,-1,-1,-1` 形式的 MOT 格式）；帧号在 `[1, seqLength]`；
- 旧自动 watcher 曾在文件名检查上翻车（shell 转义问题），现在由 Python 的 `zipfile` 直接校验，不依赖 sed。

## 4. 尚未记录、站点恢复后要确认的

1. **token 获取**：记录里的 token 是外部提供并事后删除的；`/api/api-token-auth/` 只是按 DRF 惯例的假设，未实测。
2. **公开榜单开关**：提交默认 `is_public: false`（不上公开排行榜）；旧通知里提到"可以用 API 设为 public"，但具体端点未落盘，需要恢复后用一次提交确认（很可能是对 submission 的 PATCH/POST）。
3. **MOT17 / SportsMOT 的 competition / phase / task ID**：仓库只有 MOT20 的记录。
4. 提交名称（`name` 字段）会显示在提交记录中；旧流程用 `方法名_final_post` 形式。

## 5. 对应工具

`tools/codabench_submit.py`（worktree `feature/shutrack-a-opt`，未提交）：

- `login`：`/api/api-token-auth/` 换 token（待站点恢复验证）；
- `validate`：按上面的文件集合/列数/帧范围规则校验 zip（已用历史 MOT20/MOT17 zip 实测通过）；
- `submit`：与上面三条 curl 完全同形的 reserve → PUT → create，可选 `--poll` 与 `--save-record`；`--dry-run` 只打印将要发送的 curl 命令，不发请求；
- `poll`：按 submission id 取状态与全部 scores。

站点恢复后的第一步：`login` 或直接给 token → 用历史 MOT20 zip 做一次 `--dry-run` 对照 → 经你同意后做一次真实提交，确认公开开关端点后，再接入"训练→打包→校验→提交→收集分数"的自动链路。
