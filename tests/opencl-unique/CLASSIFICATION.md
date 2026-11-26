# opencl-unique 用例分类（84 个）

- 版本：Vortex 3.0，commit `3460fa819`
- 日期：2026-11-26
- 用例总数：84（默认扫描 76 + EXCLUDE 8）
- 分类维度：**Basic / Advanced**（每个用例一个类型）

## 来源与口径说明

- 本表的 Basic / Advanced 标注，取自 2026-11-26 在 `RESULTS.md` **工作副本**中的人工标注。
  该改动已按要求还原，因此**本文档是这一分类的唯一记录**。
- 运行条件（是否默认扫描、是否需要特殊 PoCL / A 扩展）来自 `Makefile` 的 `EXCLUDE`
  列表与 `README.md`，与 `RESULTS.md` 一致。
- 全部 84 个用例在 64/32 位 × simx/rtlsim 下均已 PASS，逐用例结果见
  [`RESULTS.md`](RESULTS.md)；本文档只记录分类，不重复结果数据。

**关于分类口径（推测，未向标注者确认）：** 从现有标注归纳，Basic 多为单 kernel、
单阶段或教学型的算子/小程序（向量运算、归约、排序、单个线性代数内核、单个 NN 算子等）；
Advanced 多为多阶段、多 kernel 或数据流/控制流复杂的应用级用例（完整应用、NN 网络链、
图像/图算法/流体等）。若这一口径不对，请直接修正标注，本文档随之更新。

## 概览

| 类型 | 合计 | 默认扫描 | EXCLUDE |
|---|---|---|---|
| Basic | 38 | 32 | 6 |
| Advanced | 46 | 44 | 2 |
| **合计** | **84** | **76** | **8** |

## Basic（38）

| case | 来源 | 默认扫描 | 备注 |
|---|---|---|---|
| `atomicreduce` | Vortex 原生 | 否 | 需 `CONFIGS=-DVX_CFG_EXT_A_ENABLE` |
| `blueporcelain-bitonic-sort` | blueporcelain | 是 | |
| `blueporcelain-dot-product` | blueporcelain | 是 | |
| `blueporcelain-matrix-inversion` | blueporcelain | 是 | |
| `blueporcelain-matrix-transpose` | blueporcelain | 是 | |
| `blueporcelain-nbody` | blueporcelain | 是 | |
| `blueporcelain-nqueen` | blueporcelain | 是 | |
| `blueporcelain-reduction-max` | blueporcelain | 是 | |
| `blueporcelain-triangle-area` | blueporcelain | 是 | |
| `blueporcelain-vectoradd` | blueporcelain | 是 | |
| `blueporcelain-vectordiv` | blueporcelain | 是 | |
| `blueporcelain-vectormma` | blueporcelain | 是 | |
| `copybuf` | Vortex 原生 | 否 | 只测试 buffer copy API，不启动 kernel（`instrs=0`） |
| `image_copy` | Vortex 原生 | 否 | 需要 image-enabled 的 PoCL 构建 |
| `image_hwsw` | Vortex 原生 | 否 | 需要 image-enabled 的 PoCL 构建与 TEX/RTU |
| `image_int` | Vortex 原生 | 否 | 需要 image-enabled 的 PoCL 构建 |
| `kmeans` | Vortex 原生 | 是 | |
| `nearn` | Vortex 原生 | 是 | |
| `oclprintf` | Vortex 原生 | 是 | |
| `parboil-histo` | parboil | 否 | 需 `CONFIGS=-DVX_CFG_EXT_A_ENABLE` |
| `parboil-spmv` | parboil | 是 | |
| `polybench-3mm` | polybench | 是 | |
| `polybench-adi` | polybench | 是 | |
| `polybench-covariance` | polybench | 是 | |
| `polybench-doitgen` | polybench | 是 | |
| `polybench-gemver` | polybench | 是 | |
| `polybench-lu` | polybench | 是 | |
| `polybench-syr2k` | polybench | 是 | |
| `saxpy` | Vortex 原生 | 是 | |
| `shoc-reduction` | shoc | 是 | |
| `shoc-scan` | shoc | 是 | |
| `shoc-sort` | shoc | 是 | |
| `shoc-spmv` | shoc | 是 | |
| `shoc-stencil2d` | shoc | 是 | |
| `shoc-triad` | shoc | 是 | |
| `spmv` | Vortex 原生 | 是 | |
| `ventus-conv` | ventus | 是 | |
| `ventus-gemm` | ventus | 是 | |

## Advanced（46）

| case | 来源 | 默认扫描 | 备注 |
|---|---|---|---|
| `backprop` | Vortex 原生 | 是 | |
| `blackscholes` | Vortex 原生 | 是 | |
| `blueporcelain-nn-op-adaptiveavgpool2d` | blueporcelain | 是 | |
| `blueporcelain-nn-op-batchnorm3d` | blueporcelain | 是 | |
| `blueporcelain-nn-op-conv3d` | blueporcelain | 是 | |
| `blueporcelain-nn-op-dropout` | blueporcelain | 是 | |
| `blueporcelain-nn-op-embedding` | blueporcelain | 是 | |
| `blueporcelain-nn-op-gelu` | blueporcelain | 是 | |
| `blueporcelain-nn-op-gru` | blueporcelain | 是 | |
| `blueporcelain-nn-op-hardsigmoid` | blueporcelain | 是 | |
| `blueporcelain-nn-op-hardswish` | blueporcelain | 是 | |
| `blueporcelain-nn-op-layernorm` | blueporcelain | 是 | |
| `blueporcelain-nn-op-leakyrelu` | blueporcelain | 是 | |
| `blueporcelain-nn-op-linear` | blueporcelain | 是 | |
| `blueporcelain-nn-op-lstm` | blueporcelain | 是 | |
| `blueporcelain-nn-op-sigmoid` | blueporcelain | 是 | |
| `blueporcelain-nn-op-silu` | blueporcelain | 是 | |
| `blueporcelain-nn-op-tanh` | blueporcelain | 是 | |
| `btree` | Vortex 原生 | 是 | |
| `cfd` | Vortex 原生 | 是 | |
| `dwt2d` | Vortex 原生 | 是 | |
| `heartwall` | Vortex 原生 | 是 | |
| `hotspot3D` | Vortex 原生 | 是 | |
| `myocyte` | Vortex 原生 | 是 | |
| `nw` | Vortex 原生 | 是 | |
| `parboil-lbm` | parboil | 是 | |
| `parboil-mri-gridding` | parboil | 否 | 需 `CONFIGS=-DVX_CFG_EXT_A_ENABLE` |
| `parboil-mri-q` | parboil | 是 | |
| `parboil-sad` | parboil | 是 | |
| `parboil-tpacf` | parboil | 否 | 需 EXT_A（实测默认配置也能通过，移植不含原子内建） |
| `rodinia-leukocyte` | rodinia | 是 | |
| `rodinia-particlefilter-single` | rodinia | 是 | |
| `rodinia-srad-v2` | rodinia | 是 | |
| `shoc-bfs` | shoc | 是 | |
| `shoc-md` | shoc | 是 | |
| `shoc-md5hash` | shoc | 是 | |
| `srad` | Vortex 原生 | 是 | |
| `ventus-batchnorm2d` | ventus | 是 | |
| `ventus-conv-bn` | ventus | 是 | |
| `ventus-conv-fused` | ventus | 是 | |
| `ventus-gemm-fused` | ventus | 是 | |
| `ventus-mnist` | ventus | 是 | |
| `ventus-mnist-conv-small` | ventus | 是 | |
| `ventus-pool2d` | ventus | 是 | |
| `ventus-relu` | ventus | 是 | |
| `ventus-softmax` | ventus | 是 | |

## 按来源分布

| 来源 | 合计 | Basic | Advanced |
|---|---|---|---|
| blueporcelain | 27 | 11 | 16 |
| polybench | 7 | 7 | 0 |
| rodinia | 3 | 0 | 3 |
| parboil | 7 | 2 | 5 |
| shoc | 9 | 6 | 3 |
| ventus | 11 | 2 | 9 |
| Vortex 原生（无前缀） | 20 | 10 | 10 |
| **合计** | **84** | **38** | **46** |

> 说明：`来源` 按用例名前缀判定；无前缀者为 `tests/opencl` 中的 Vortex 原生用例。
> 同名算法在不同来源中可能重复出现（例如 `shoc-spmv` / `parboil-spmv` / `spmv`、
> `srad` / `rodinia-srad-v2`），本表按用例记录，不做合并。