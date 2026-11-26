# Vortex 2.2 — opencl-unique 测试结果

- 版本：Vortex 2.2，commit `e86eeab8e`
- 日期：2026-11-25
- 构建：`build/`（XLEN=64）、`build32/`（XLEN=32）；驱动：simx、rtlsim
- 用例：TESTS 84，默认扫描 69，EXCLUDE 15
- 用例名与顺序与 3.0 的 `tests/opencl-unique` 完全一致（84/84）

## 默认扫描结果

| case | 64 simx | 64 rtlsim | 32 simx | 32 rtlsim | instrs 64 (s/r) | instrs 32 (s/r) |
|---|---|---|---|---|---|---|
| `backprop` | PASS | PASS | PASS | PASS | 9404/9406 | 8755/8756 |
| `blackscholes` | PASS | PASS | PASS | PASS | 108882/108885 | 3124114/3124118 |
| `blueporcelain-bitonic-sort` | PASS | PASS | PASS | PASS | 68467/68469 | 66346/66347 |
| `blueporcelain-dot-product` | PASS | PASS | PASS | PASS | 88310/88313 | 85163/85167 |
| `blueporcelain-matrix-inversion` | PASS | PASS | PASS | PASS | 62016/62018 | 59413/59415 |
| `blueporcelain-matrix-transpose` | PASS | PASS | PASS | PASS | 807293/807296 | 790322/790326 |
| `blueporcelain-nbody` | PASS | PASS | PASS | PASS | 248886/248889 | 238379/238383 |
| `blueporcelain-nn-op-adaptiveavgpool2d` | PASS | PASS | PASS | PASS | 5064/5066 | 4530/4531 |
| `blueporcelain-nn-op-batchnorm3d` | PASS | PASS | PASS | PASS | 4949/4952 | 4411/4413 |
| `blueporcelain-nn-op-conv3d` | PASS | PASS | PASS | PASS | 26429/26432 | 26108/26112 |
| `blueporcelain-nn-op-dropout` | PASS | PASS | PASS | PASS | 9075/9078 | 8360/8364 |
| `blueporcelain-nn-op-embedding` | PASS | PASS | PASS | PASS | 11510/11512 | 10924/10926 |
| `blueporcelain-nn-op-gelu` | PASS | PASS | PASS | PASS | 15407/15410 | 202306/202310 |
| `blueporcelain-nn-op-gru` | PASS | PASS | PASS | PASS | 9213/9215 | 54794/54798 |
| `blueporcelain-nn-op-hardsigmoid` | PASS | PASS | PASS | PASS | 7862/7864 | 7149/7150 |
| `blueporcelain-nn-op-hardswish` | PASS | PASS | PASS | PASS | 7926/7928 | 7213/7214 |
| `blueporcelain-nn-op-layernorm` | PASS | PASS | PASS | PASS | 4711/4714 | 4094/4095 |
| `blueporcelain-nn-op-leakyrelu` | PASS | PASS | PASS | PASS | 7839/7841 | 7126/7127 |
| `blueporcelain-nn-op-linear` | PASS | PASS | PASS | PASS | 6640/6643 | 5700/5704 |
| `blueporcelain-nn-op-lstm` | PASS | PASS | PASS | PASS | 11550/11553 | 80470/80474 |
| `blueporcelain-nn-op-sigmoid` | PASS | PASS | PASS | PASS | 12149/12152 | 198981/198985 |
| `blueporcelain-nn-op-silu` | PASS | PASS | PASS | PASS | 12277/12280 | 199109/199113 |
| `blueporcelain-nn-op-tanh` | PASS | PASS | PASS | PASS | 12341/12344 | 199332/199336 |
| `blueporcelain-reduction-max` | PASS | PASS | PASS | PASS | 139492/139494 | 128155/128156 |
| `blueporcelain-triangle-area` | PASS | PASS | PASS | PASS | 21915/21918 | 20818/20820 |
| `blueporcelain-vectoradd` | PASS | PASS | PASS | PASS | 16767/16769 | 17203/17207 |
| `blueporcelain-vectordiv` | PASS | PASS | PASS | PASS | 16758/16761 | 15659/15663 |
| `blueporcelain-vectormma` | PASS | PASS | PASS | PASS | 31094/31097 | 29486/29487 |
| `dwt2d` | PASS | PASS | PASS | PASS | 10730/10732 | 9773/9777 |
| `heartwall` | PASS | PASS | PASS | PASS | 227268/227271 | 207753/207757 |
| `hotspot3D` | PASS | PASS | PASS | PASS | 64637/64640 | 68914/68918 |
| `kmeans` | PASS | PASS | PASS | PASS | -/- | -/- |
| `myocyte` | PASS | PASS | PASS | PASS | 11407/11410 | 255544/255545 |
| `nearn` | PASS | PASS | PASS | PASS | 1139613/1139615 | 1096201/1096202 |
| `oclprintf` | PASS | PASS | PASS | PASS | 7416/7419 | 6742/6744 |
| `parboil-lbm` | PASS | PASS | PASS | PASS | 160678/160681 | 147293/147295 |
| `parboil-mri-q` | PASS | PASS | PASS | PASS | 905941/905944 | 24093419/24093421 |
| `parboil-sad` | PASS | PASS | PASS | PASS | 153360/153363 | 118789/118793 |
| `parboil-spmv` | PASS | PASS | PASS | PASS | 37750/37753 | 36651/36655 |
| `polybench-3mm` | PASS | PASS | PASS | PASS | 368005/368008 | 295738/295742 |
| `polybench-adi` | PASS | PASS | PASS | PASS | 5451/5454 | 4865/4867 |
| `polybench-covariance` | PASS | PASS | PASS | PASS | 973054/973057 | 972214/972215 |
| `polybench-doitgen` | PASS | PASS | PASS | PASS | 13267/13270 | 12392/12396 |
| `polybench-gemver` | PASS | PASS | PASS | PASS | 36415/36417 | 35702/35703 |
| `polybench-lu` | PASS | PASS | PASS | PASS | 17795/17798 | 16696/16700 |
| `polybench-syr2k` | PASS | PASS | PASS | PASS | 602494/602496 | 530226/530230 |
| `rodinia-leukocyte` | PASS | PASS | PASS | PASS | 14834294/14834296 | 15023083/15023087 |
| `rodinia-particlefilter-single` | PASS | PASS | PASS | PASS | 8971/8974 | 8335/8339 |
| `saxpy` | PASS | PASS | PASS | PASS | 48252/48255 | 44646/44650 |
| `shoc-bfs` | PASS | PASS | PASS | PASS | 132235/132238 | 132930/132931 |
| `shoc-md` | PASS | PASS | PASS | PASS | 3231233/3231233 | 2836403/2836407 |
| `shoc-md5hash` | PASS | PASS | PASS | PASS | 2915725/2915727 | 2910532/2910533 |
| `shoc-reduction` | PASS | PASS | PASS | PASS | 164918/164921 | 160363/160367 |
| `shoc-scan` | PASS | PASS | PASS | PASS | 65374/65377 | 60947/60951 |
| `shoc-spmv` | PASS | PASS | PASS | PASS | 28258/28261 | 26157/26159 |
| `shoc-stencil2d` | PASS | PASS | PASS | PASS | 346052/346054 | 291256/291260 |
| `shoc-triad` | PASS | PASS | PASS | PASS | 205174/205177 | 196395/196399 |
| `spmv` | PASS | PASS | PASS | PASS | 150566/150569 | 147728/147732 |
| `srad` | PASS | PASS | PASS | PASS | 129406/129409 | 3175666/3175668 |
| `ventus-batchnorm2d` | PASS | PASS | PASS | PASS | 39808/39811 | 38453/38457 |
| `ventus-conv` | PASS | PASS | PASS | PASS | 1689688/1689691 | 1634349/1634353 |
| `ventus-conv-bn` | PASS | PASS | PASS | PASS | 1727456/1727459 | 1672117/1672121 |
| `ventus-conv-fused` | PASS | PASS | PASS | PASS | 1591852/1591855 | 1614273/1614277 |
| `ventus-gemm` | PASS | PASS | PASS | PASS | 230781/230784 | 192306/192310 |
| `ventus-gemm-fused` | PASS | PASS | PASS | PASS | 238479/238481 | 197955/197959 |
| `ventus-mnist` | PASS | PASS | PASS | PASS | 16889/16892 | 12411/12413 |
| `ventus-mnist-conv-small` | PASS | PASS | PASS | PASS | 16920/16923 | 15406/15407 |
| `ventus-pool2d` | PASS | PASS | PASS | PASS | 45309/45311 | 46363/46367 |
| `ventus-relu` | PASS | PASS | PASS | PASS | 32129/32131 | 30775/30777 |

## 特殊配置用例（已验证通过，不属于问题）

| case | 说明 | 64 simx | 64 rtlsim | 32 simx | 32 rtlsim |
|---|---|---|---|---|---|
| `parboil-tpacf` | 带 EXT_A runtime 通过；其移植不含原子内建，默认 runtime 也能通过 | PASS | PASS | PASS | PASS |

## 本版本有问题的测试

| case | 类别 | 原因 |
|---|---|---|
| `atomicreduce` | 工具链限制 | 同上，缺少 `_cl_atomic_add` |
| `blueporcelain-nqueen` | 模型不兼容 | 2.2 simx 发散中止；rtlsim 结果错 1（92 vs 82） |
| `btree` | 模型不兼容 | 2.2 simx 卡死；rtlsim 调度器 stall timeout（死锁） |
| `cfd` | 运行失败 | 2.2 PoCL clEnqueueCopyBuffer 路径 NULL 调用 |
| `copybuf` | 运行失败 | 2.2 PoCL clEnqueueCopyBuffer 路径 NULL 调用 |
| `image_copy` | 能力缺失 | 2.2 PoCL 报 no image support |
| `image_hwsw` | 能力缺失 | 2.2 无 image 支持，且无 TEX/RTU 单元 |
| `image_int` | 能力缺失 | 2.2 PoCL 报 no image support |
| `nw` | 模型不兼容 | 2.2 32 位 simx 结果错误；64 位 simx 与 32 位 rtlsim 均 PASS |
| `parboil-histo` | 工具链限制 | 2.2 PoCL 以 OpenCL C 1.2 编译，内核原子内建缺少 `_cl_atomic_add/min/max` |
| `parboil-mri-gridding` | 工具链限制 | 同上，缺少 `_cl_atomic_add/_sub`（链接失败） |
| `rodinia-srad-v2` | 模型不兼容 | 2.2 64 位 simx 结果错误；32 位 simx 与 64 位 rtlsim 均 PASS |
| `shoc-sort` | 模型不兼容 | 2.2 simx 卡死；rtlsim 调度器 stall timeout（死锁） |
| `ventus-softmax` | 模型不兼容 | 2.2 simx 结果错误（归一化因子=最大指数项）；rtlsim PASS |

## 失败原因分析（能力缺失/原子之外）

| case | simx | rtlsim | 证据与结论 | 置信度 |
|---|---|---|---|---|
| `blueporcelain-nqueen` | divergent branch! PC=0x1800001b0 并 abort | FAILED! - 1（期望 92，实际 82） | 2.2 simx 的分支检查同样要求 warp 活跃 lane 一致；nqueen 的 while(i>=0) 回溯循环天然发散。 2.2 simx 发散限制 + RTL 发散执行错误；3.0 双驱动通过。 | 高 |
| `btree` | 卡死 | VX_schedule.sv:435 stall timeout 断言（cycle 426271） | 同 shoc-sort；work-group=16，barrier 在统一 for(height) 循环。 2.2 调度/RTL 死锁；3.0 通过。 | 中 |
| `cfd` | Segmentation fault | Segmentation fault | gdb 栈同 2.3（clEnqueueCopyBuffer -> PoCL Vortex -> NULL）。 2.2 PoCL CopyBuffer 路径 NULL 回调。 | 高 |
| `copybuf` | Segmentation fault | Segmentation fault | gdb 栈与 cfd 相同（test_basic_copy -> clEnqueueCopyBuffer）。 同一 PoCL/runtime CopyBuffer bug。 | 高 |
| `nw` | 32 位 908 errors（局部恒定偏移）；64 位 PASS | 32 位 PASS；64 位未单独复测 | 32 位 rtlsim 正确、64 位 simx 正确；错误呈现区域恒定偏移（如 expected=-32 vs actual=-40），像 DP 递推中的系统性偏差。 2.2 32 位 simx 的整数/DP 执行 bug；测试有效。 | 高（对照充分；具体指令未定位） |
| `rodinia-srad-v2` | 64 位 924 errors（差异约 2–5%）；32 位 PASS | 64 位 PASS；32 位未单独复测（调度器无关） | 同一配置 32 位 simx 与 64 位 rtlsim 均正确；错误为小范围浮点偏差（如 1.379436 vs 1.412673）。 2.2 64 位 simx 的 FPU/指令模型 bug；测试有效。 | 高（对照充分；具体指令未定位） |
| `shoc-sort` | 卡死 | VX_schedule.sv:435 stall timeout 断言（cycle 1026535） | 同 2.3：RTL stall timeout；barrier 位于统一循环；NUM_BARRIERS=8 实验未改变（在 2.3 上验证，2.2 同源）。 2.2 调度/RTL 死锁；3.0 通过。 | 中 |
| `ventus-softmax` | rows 结果错误（实际值=e_j/e_max）；chw 结果错误或卡死 | PASS（64/32 位） | 与 2.3 相同的数值规律（实际最大值=1.0）；同一 ELF 在 rtlsim 正确。 2.2 simx 执行/FP 语义 bug；测试有效。 | 高 |

## 结论与边界

- TESTS 的 84 个用例名称与顺序与 3.0 的 tests/opencl-unique 完全一致。
- 最终验证按 64 simx -> 64 rtlsim -> 32 simx -> 32 rtlsim 串行执行：2.2 的 kernel/scripts/vxbin.py 把临时文件硬编码为 /tmp/temp_kernel.bin，两个 build 树并发会互相覆盖，产生假失败（vectormma/shoc-triad/triangle-area 已确认）。
- 默认扫描 69 个；详细逐用例结果见 RESULTS.md。
- 2.2 的 simx/rtlsim instrs 不要求相等；结果中保留两侧数值供参考。
- 已重建 EXT_A runtime 并补跑 4 个原子用例；tpacf 通过，其余三个因 OpenCL 1.2 内核库缺少 _cl_atomic_* 而构建失败。
- image_* 在 2.2 上明确不受支持，归入本版本的问题清单。
- 失败原因分析：nqueen/sort/btree/softmax 与 2.3 同类；srad-v2 为 2.2 64 位 simx FP 模型 bug；nw 为 2.2 32 位 simx 整数/DP 模型 bug；cfd/copybuf 为 PoCL CopyBuffer NULL 调用。详见 diagnosis。

原始数据：`results/v2.2.json`；每个用例的完整日志未入库。
