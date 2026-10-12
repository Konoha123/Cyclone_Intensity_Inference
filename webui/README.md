# Cyclone Intensity WebUI（展示工具）

本地 Web 展示工具：上传热带气旋卫星图像 → 界面预览 → 展示 Dvorak T 值
类别概率分布（柱状图 + 概率分布折线）与文本栏指标（`t_mode` 最大概率类别
T 值、`t_mean = Σ p_i × T_i` 期望平均 T 值）。

本目录（`webui/`）自成一体：后端代码在 `backend/`，前端代码在 `frontend/`，
测试在 `tests/`；共享推理核心在仓库 `src/cyclone_intensity/`（不在本目录内）。
整体删除 `webui/` 即可完全移除本工具，不影响研究主线。

## 边界与口径（固定标注，勿移除）

- 概率为 Softmax 原始输出，**未校准**（`calibrated=false`）；
- 预处理为 Ultralytics 分类默认管线，**未核验**
  （`preprocessing_id=ultralytics-default-classify-unverified`）；
- 输出单位为 **Dvorak T-number**，不换算、不代表最大持续风速；
- 权重内实际类别表为 13 类（T1.5、T2.5–T8.0，缺 T2.0），与上游自述的
  14 类不符，界面在“类别表提示”中如实展示。

## 启动

```bash
# 在 cyclone-intensity-inference-env 环境中，仓库根目录下执行
python webui/run.py                # 默认 127.0.0.1:8000，加载 weights/model.pt
python webui/run.py --port 8123    # 自定义端口/权重/监听地址见 --help
```

浏览器打开 http://127.0.0.1:8000 。启动时校验权重 SHA-256 与类别表，
失败即拒绝启动。交互式 API 文档在 `/docs`。

## 依赖

- 后端 Python 依赖：`webui/requirements.txt`（官方安装方式
  `pip install "fastapi[standard]"` 后按 `pip freeze` 差集精确锁定）；
  另需环境内既有的 torch / ultralytics / Pillow（版本见根 README 环境登记）；
- 前端无构建步骤、无外部网络请求；第三方 JS 见下表，本地 vendor 引入。

## 前端第三方资产溯源

| 文件 | 版本 | 许可 | 大小 | SHA-256 |
| --- | --- | --- | --- | --- |
| `frontend/vendor/chart.umd.js` | Chart.js v4.5.1 | MIT | 208518 B | `ecc3cd1eeb8c34d2178e3f59fd63ec5a3d84358c11730af0b9958dc886d7652a` |

来源 URL（Chart.js 官方文档列出的 jsDelivr 渠道）：

```
https://cdn.jsdelivr.net/npm/chart.js@4.5.1/dist/chart.umd.js
```

完整性核验（2026-10-12）：jsDelivr 下载文件与 npm 官方包
`registry.npmjs.org/chart.js/-/chart.js-4.5.1.tgz` 内 `package/dist/chart.umd.js`
SHA-256 逐位一致（双官方渠道交叉验证）。复算命令：

```bash
sha256sum webui/frontend/vendor/chart.umd.js
```

本机网络备注：jsDelivr 下载在本环境需经代理（`http_proxy=127.0.0.1:7890`），
npm registry 可直连；两者结果一致。
