# Codabench API 提交链路（2026-11-18 实测更新）

站点已恢复（HTTP 200），前端新增 Anubis v1.28.2 反爬网关。本文记录实测通过的认证方式、参数格式与坑位，取代 `codabench_submission_flow_20261117.md` 中与旧版 API 相关的部分（旧文档保留作历史记录）。

## 1. 认证

- 登录端点 `/api/api-token-auth/` 使用 DRF 原生 `obtain_auth_token`，返回 40 位 DRF Token。**Token 无过期时间**，重复登录返回同一个 token；改密码/退出网页不影响。token 保存在 `~/.config/codabench/token`（0600）。
- Anubis 网关规则（实测）：
  - 无 `Authorization` 头的请求可能收到挑战页（HTML "Making sure you're not a bot!"）；
  - **带 `Authorization: Token <token>` 的请求直接放行到 Django API**；
  - `/api/submissions/` 路径连无认证 GET 也放行。
- 因此日常自动化只需 token，不需要浏览器或 Anubis cookie。Anubis cookie（JWT，7 天）只在需要重新登录时用一次。

## 2. 比赛 ID（2026-11-18 查询）

| 数据集 | competition | phase | task UUID | **task instance id（提交用）** |
|---|---|---|---|---|
| MOT17 | 10049 | 16382 | c4113629-732e-41bf-9a29-ab837ba1218c | **20803** |
| MOT20 | 10050 | 16383 | 08d6c1f3-6203-4067-824f-f72c30130693 | **20804** |
| SportsMOT | 13077 | 21402 | cfaf2a08-96f7-4602-a04b-3a41333a8cd9 | **26486** |

- v1.37.1 的创建接口要求 `tasks` 传**整数主键**（`phases[].tasks[].id`）；传 UUID 会报 `Incorrect type. Expected pk value, received str.`
- 配额：三个 phase 均为每参与者每天 1 次（`max_submissions_per_day: 1`），可用 `GET /api/can_make_submission/<phase_id>/` 查询。

## 3. 提交步骤（实测）

1. 预留：`POST /api/datasets/`（multipart）
   - `file_size=<zip 字节数>`、`name=<提交名称>`、`type=submission`、`request_sassy_file_name=<zip 文件名>`
   - **name 必须唯一**：同名再次预留会返回校验错误（没有 `key` 字段）。
   - 返回 `{"key": "<uuid>", "sassy_url": "..."}`。
2. 上传：`PUT <sassy_url>`，`Content-Type: application/zip`，二进制 body → 200。
3. 创建：`POST /api/submissions/`（multipart）
   - `data=<key>`、`phase=<phase id>`、`tasks=<task instance id 整数>`
   - 返回 `{"id": <submission id>, "status": "Submitting", "filename": "...", ...}`。
   - `filename` 在记录中会被截断到约 40 字符；用于对比时以预留时的 `name` 为准。
4. 轮询：`GET /api/submissions/<id>/`，30 s 间隔；状态 `Submitting → Running → Finished`（失败为 `Failed`/`Cancelled`）。评分约 10–20 分钟。
5. 收集：完成后 `scores` 为 `[{column_key, score, is_primary}]`，score 为百分比字符串。

## 4. 本轮提交记录（2026-11-18）

| 数据集 | submission id | 预留 name | ZIP SHA-256 |
|---|---|---|---|
| MOT20 | 1057845 | shutrackA_mot20_sqrtsize_e100_post_20261118_v2 | b7bd4a40…a51cbae3 |
| MOT17 | 1057846 | shutrackA_mot17_fromM20interleaved_e50_post_20261118 | 18f9a5b1…001d0a42 |
| SportsMOT | 1057848 | shutrackA_sportsmot_trainval_e200_post_20261118 | d35720b5…93ee3283 |

- MOT20 首次创建因 `tasks` 传 UUID 失败，产生一个未绑定提交的预留数据 `b7ae4da5-dd34-4c37-8236-6c618773fc02`（已上传但无 submission）；本轮 MOT20 实际使用 `_v2` 名称的预留 `a5b5c590-b2b1-4c70-9f0e-f90219803cec`。该孤儿数据不占配额，可在账号数据页手动删除。
- 创建记录：`logs/submission_<dataset>_create.json`；最终分数：`logs/submission_<dataset>_<id>_final.json`；汇总：`logs/submission_<dataset>_<id>_summary.json`。

## 5. 规则提醒

MOT17/MOT20 的 Participation 页明确：评测服务器不得用于训练/调参，提交后需等 24 小时，每个方法只向测试服务器提交一个结果。平台的“每账号每天 1 次”即该规则的执行。多账号轮换提交属于规避行为，有成绩作废/封号风险；如需高频测试，应联系组织者申请（codabench@motchallenge.net）。
