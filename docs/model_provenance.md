# 模型权重与示例数据来源清单（验证记录）

> 记录外部获取的权重与图像的来源、哈希与核验结果。依据 `AGENTS.md` 第 4.1、5.2 节：外部文件不进 Git，但来源清单必须可核对。首次记录日期：2026-10-10。

## 1. 来源仓库

| 项目 | 值 |
| --- | --- |
| 上游仓库 | <https://github.com/EnceladusCat/Cyclone-Intensity-ML>（ENM-AIIR / AGCI） |
| 下载时 main 分支 HEAD | `55328c4d43b0570dea2d986c08d89b25e5546711`（2026-09-26，GitHub 签名提交 "Update README.md"） |
| 上游许可证 | MIT（Copyright (c) 2026 MFI）；不自动覆盖本仓库其他组件 |
| 在线展示 | <https://enm-aiir.pages.dev/> |
| 下载方式 | `curl -L` 自 `raw.githubusercontent.com`（main 分支），2026-10-10 |

## 2. 本地文件与哈希核验

### 2.1 模型权重

| 项目 | 值 |
| --- | --- |
| 本地路径 | `weights/model.pt`（Git 忽略） |
| 上游路径 | 仓库根目录 `model.pt` |
| 大小 | 94,973,800 字节 |
| SHA-256 | `55fd4f72af0e3825dccdd4dcccb82a8d8b46739a30b3681304c35528f8e078a1` |
| Git blob SHA | `609ee818ecc616a77aa0e5801b5fb5f669f387cc` |
| 与上游一致性 | **已核验**：本地 `git hash-object` 结果、字节数与 GitHub Contents API（ref=main）返回值一致 |
| 文件格式 | Zip archive（PyTorch checkpoint 容器，`file` 实测） |

### 2.2 示例图像

| 项目 | 值 |
| --- | --- |
| 本地路径 | `data/samples/example_image.jpg`（Git 忽略） |
| 上游路径 | 仓库根目录 `example_image.jpg` |
| 大小 | 28,810 字节 |
| SHA-256 | `725baa6e31e0e1da47b15578900afa73cc97ab5dbab9826a68847130ac6c0f3b` |
| Git blob SHA | `1c6c6cb6431a8dc99f2f4ceed9ad3539214667a2` |
| 与上游一致性 | **已核验**：同上 |
| 文件格式 | JPEG，baseline，512×512，单通道（灰度），`file` 实测 |

### 2.3 上游自报验证结果图（非输入样本）

| 项目 | 值 |
| --- | --- |
| 本地路径 | `data/reference/valid_test.png`（Git 忽略） |
| 上游路径 | 仓库根目录 `valid_test.png` |
| 大小 | 70,109 字节 |
| SHA-256 | `8bc256cdb3ed03ca39f4a6cbcd6eb3e71795ef735fac038943440e81aafcf019` |
| Git blob SHA | `cdc825fafe7be62c91e412c0d4cda4fcbbc41801` |
| 与上游一致性 | **已核验**：本地 `git hash-object` 结果、字节数与 GitHub Contents API（ref=main）返回值一致（2026-10-10） |
| 文件格式 | PNG，700×600，8-bit RGBA，`file` 实测 |

性质说明：此图为**作者自报验证集结果的图表**，仅作参考对照材料，不是模型输入样本，也不构成本项目的独立评估证据（见第 3 节）。

## 3. 上游自述的模型约定（未实测，仅转录）

以下为上游 README 描述，**本项目尚未核验**，不作为已确认的接口契约：

- 架构：YOLOv8m-cls 骨干；分类头按 T 值区间 T1.5–T8.0（步长 0.5，共 14 类）训练；推理用类别概率加权平均得到连续 T 值。
- 输入：512×512，单波段长波红外（IR Band-13）。
- 内部类别索引：T 值 ×10（上游自述）。
- 训练域：西北太平洋与澳洲盆地，3,100+/3,600+ 张再分析标注图像（中英文描述数量不一致，以英文 3,600+ 为准存疑）。
- 作者自报指标（保留验证集，T2.5–T8.0）：RMSE 0.31 / MAE 0.24 / ±0.5T 准确率 90.40%——**属上游自报，非本项目独立结果**。

## 4. 待验证事项（承接 AGENTS.md 第 9 节）

| 事项 | 状态 |
| --- | --- |
| `model.names` 与 14 类 T 值映射 | 未验证（加载权重后核验） |
| 输入通道数 / 预处理（CenterCrop、缩放、归一化） | 未验证 |
| 加载兼容性（ultralytics 8.4.175 + torch 2.14.1+cpu） | 未验证 |
| 单图推理 Smoke Test | 未执行 |

## 5. 复算命令

```bash
sha256sum weights/model.pt data/samples/example_image.jpg
git hash-object weights/model.pt data/samples/example_image.jpg
stat -c '%n %s bytes' weights/model.pt data/samples/example_image.jpg
curl -s https://api.github.com/repos/EnceladusCat/Cyclone-Intensity-ML/contents/model.pt?ref=main
```

## 6. 变更历史

| 日期 | 变更 |
| --- | --- |
| 2026-10-10 | 首次下载并记录；建立 `data/samples|raw|processed` 子目录结构；核验与上游一致性 |
| 2026-10-10 | 补充下载上游 `valid_test.png`（自报验证结果图）至 `data/reference/`，核验一致性并登记（2.3 节）；建立 `data/reference/` 子目录，data 目录规则见 `docs/data_layout.md` |
