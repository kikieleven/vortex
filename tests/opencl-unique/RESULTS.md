# Vortex 2.3 — opencl-unique 测试结果

- 版本：Vortex 2.3，commit `d425a56ef`
- 日期：2026-11-25
- 构建：`build/`（XLEN=64）、`build32/`（XLEN=32）；驱动：simx、rtlsim
- 用例：TESTS 84，默认扫描 71，EXCLUDE 13
- 用例名与顺序与 3.0 的 `tests/opencl-unique` 完全一致（84/84）

## 默认扫描结果

| case | 64 simx | 64 rtlsim | 32 simx | 32 rtlsim | instrs 64 (s/r) | instrs 32 (s/r) |
|---|---|---|---|---|---|---|
| `backprop` | PASS | PASS | PASS | PASS | 9199/9201 | 8593/8595 |
| `blackscholes` | PASS | PASS | PASS | PASS | 108434/108434 | 3123677/3123680 |
| `blueporcelain-bitonic-sort` | PASS | PASS | PASS | PASS | 67686/67688 | 65610/65610 |
| `blueporcelain-dot-product` | PASS | PASS | PASS | PASS | 86762/86764 | 83660/83662 |
| `blueporcelain-matrix-inversion` | PASS | PASS | PASS | PASS | 58931/58933 | 56372/56374 |
| `blueporcelain-matrix-transpose` | PASS | PASS | PASS | PASS | 782706/782707 | 765779/765781 |
| `blueporcelain-nbody` | PASS | PASS | PASS | PASS | 245803/245804 | 235340/235342 |
| `blueporcelain-nn-op-adaptiveavgpool2d` | PASS | PASS | PASS | PASS | 4971/4973 | 4482/4482 |
| `blueporcelain-nn-op-batchnorm3d` | PASS | PASS | PASS | PASS | 4833/4835 | 4338/4340 |
| `blueporcelain-nn-op-conv3d` | PASS | PASS | PASS | PASS | 26250/26251 | 25973/25975 |
| `blueporcelain-nn-op-dropout` | PASS | PASS | PASS | PASS | 8871/8873 | 8201/8203 |
| `blueporcelain-nn-op-embedding` | PASS | PASS | PASS | PASS | 11113/11115 | 10571/10573 |
| `blueporcelain-nn-op-gelu` | PASS | PASS | PASS | PASS | 15205/15205 | 202147/202149 |
| `blueporcelain-nn-op-gru` | PASS | PASS | PASS | PASS | 9120/9122 | 54747/54749 |
| `blueporcelain-nn-op-hardsigmoid` | PASS | PASS | PASS | PASS | 7657/7659 | 6988/6989 |
| `blueporcelain-nn-op-hardswish` | PASS | PASS | PASS | PASS | 7721/7723 | 7052/7053 |
| `blueporcelain-nn-op-layernorm` | PASS | PASS | PASS | PASS | 4603/4605 | 4029/4030 |
| `blueporcelain-nn-op-leakyrelu` | PASS | PASS | PASS | PASS | 7634/7636 | 6965/6966 |
| `blueporcelain-nn-op-linear` | PASS | PASS | PASS | PASS | 6580/6582 | 5685/5688 |
| `blueporcelain-nn-op-lstm` | PASS | PASS | PASS | PASS | 11458/11460 | 80423/80425 |
| `blueporcelain-nn-op-sigmoid` | PASS | PASS | PASS | PASS | 11945/11947 | 198822/198824 |
| `blueporcelain-nn-op-silu` | PASS | PASS | PASS | PASS | 12073/12075 | 198950/198952 |
| `blueporcelain-nn-op-tanh` | PASS | PASS | PASS | PASS | 12137/12140 | 199173/199175 |
| `blueporcelain-reduction-max` | PASS | PASS | PASS | PASS | 136407/136409 | 125114/125115 |
| `blueporcelain-triangle-area` | PASS | PASS | PASS | PASS | 21135/21137 | 20081/20083 |
| `blueporcelain-vectoradd` | PASS | PASS | PASS | PASS | 15986/15988 | 16468/16470 |
| `blueporcelain-vectordiv` | PASS | PASS | PASS | PASS | 15980/15980 | 14924/14926 |
| `blueporcelain-vectormma` | PASS | PASS | PASS | PASS | 29546/29548 | 27982/27982 |
| `dwt2d` | PASS | PASS | PASS | PASS | 10669/10671 | 9758/9760 |
| `heartwall` | PASS | PASS | PASS | PASS | 225192/225195 | 205722/205724 |
| `hotspot3D` | PASS | PASS | PASS | PASS | 63857/63859 | 68179/68182 |
| `kmeans` | PASS | PASS | PASS | PASS | -/- | -/- |
| `myocyte` | PASS | PASS | PASS | PASS | 11299/11301 | 255480/255480 |
| `nearn` | PASS | PASS | PASS | PASS | 1075226/1075228 | 1031826/1031827 |
| `nw` | PASS | PASS | PASS | PASS | 21769/21771 | 18225/18228 |
| `oclprintf` | PASS | PASS | PASS | PASS | 7307/7309 | 6661/6663 |
| `parboil-lbm` | PASS | PASS | PASS | PASS | 159130/159133 | 145788/145790 |
| `parboil-mri-q` | PASS | PASS | PASS | PASS | 905881/905883 | 24093402/24093404 |
| `parboil-sad` | PASS | PASS | PASS | PASS | 153156/153158 | 118630/118632 |
| `parboil-spmv` | PASS | PASS | PASS | PASS | 36971/36972 | 35916/35918 |
| `polybench-3mm` | PASS | PASS | PASS | PASS | 364921/364923 | 292699/292701 |
| `polybench-adi` | PASS | PASS | PASS | PASS | 5391/5393 | 4848/4850 |
| `polybench-covariance` | PASS | PASS | PASS | PASS | 972850/972852 | 972052/972054 |
| `polybench-doitgen` | PASS | PASS | PASS | PASS | 12823/12825 | 11993/11995 |
| `polybench-gemver` | PASS | PASS | PASS | PASS | 36210/36212 | 35540/35542 |
| `polybench-lu` | PASS | PASS | PASS | PASS | 17015/17018 | 15961/15963 |
| `polybench-syr2k` | PASS | PASS | PASS | PASS | 599409/599411 | 527187/527189 |
| `rodinia-leukocyte` | PASS | PASS | PASS | PASS | 14834233/14834235 | 15023068/15023070 |
| `rodinia-particlefilter-single` | PASS | PASS | PASS | PASS | 8911/8913 | 8320/8322 |
| `rodinia-srad-v2` | PASS | PASS | PASS | PASS | 182647/182650 | 180057/180059 |
| `saxpy` | PASS | PASS | PASS | PASS | 45114/45116 | 41521/41523 |
| `shoc-bfs` | PASS | PASS | PASS | PASS | 131455/131458 | 132192/132194 |
| `shoc-md` | PASS | PASS | PASS | PASS | 3228146/3228148 | 2833364/2833367 |
| `shoc-md5hash` | PASS | PASS | PASS | PASS | 2914944/2914946 | 2909795/2909796 |
| `shoc-reduction` | PASS | PASS | PASS | PASS | 161836/161836 | 157324/157326 |
| `shoc-scan` | PASS | PASS | PASS | PASS | 64594/64596 | 60212/60214 |
| `shoc-spmv` | PASS | PASS | PASS | PASS | 27478/27480 | 25420/25422 |
| `shoc-stencil2d` | PASS | PASS | PASS | PASS | 342967/342969 | 288217/288220 |
| `shoc-triad` | PASS | PASS | PASS | PASS | 192875/192876 | 184140/184142 |
| `spmv` | PASS | PASS | PASS | PASS | 147045/147046 | 144219/144221 |
| `srad` | PASS | PASS | PASS | PASS | 126322/126324 | 3172625/3172627 |
| `ventus-batchnorm2d` | PASS | PASS | PASS | PASS | 38644/38646 | 37334/37336 |
| `ventus-conv` | PASS | PASS | PASS | PASS | 1683533/1683534 | 1628238/1628240 |
| `ventus-conv-bn` | PASS | PASS | PASS | PASS | 1721301/1721302 | 1666006/1666008 |
| `ventus-conv-fused` | PASS | PASS | PASS | PASS | 1585696/1585698 | 1608162/1608164 |
| `ventus-gemm` | PASS | PASS | PASS | PASS | 227697/227700 | 189267/189269 |
| `ventus-gemm-fused` | PASS | PASS | PASS | PASS | 235394/235396 | 194916/194918 |
| `ventus-mnist` | PASS | PASS | PASS | PASS | 16829/16831 | 12394/12396 |
| `ventus-mnist-conv-small` | PASS | PASS | PASS | PASS | 16428/16430 | 14956/14958 |
| `ventus-pool2d` | PASS | PASS | PASS | PASS | 44912/44914 | 46012/46014 |
| `ventus-relu` | PASS | PASS | PASS | PASS | 30964/30966 | 29654/29656 |

## 特殊配置用例（已验证通过，不属于问题）

| case | 说明 | 64 simx | 64 rtlsim | 32 simx | 32 rtlsim |
|---|---|---|---|---|---|
| `parboil-tpacf` | 带 EXT_A runtime 通过；其移植不含原子内建，默认 runtime 也能通过 | PASS | PASS | PASS | PASS |

## 本版本有问题的测试

| case | 类别 | 原因 |
|---|---|---|
| `atomicreduce` | 工具链限制 | 同上，缺少 `_cl_atomic_add` |
| `blueporcelain-nqueen` | 模型不兼容 | 2.3 simx 发散中止；rtlsim 结果错 1（92 vs 82），2.3 模型 bug |
| `btree` | 模型不兼容 | 2.3 simx 卡死；rtlsim 调度器 stall timeout（死锁） |
| `cfd` | 运行失败 | 2.3 PoCL clEnqueueCopyBuffer 路径 NULL 调用 |
| `copybuf` | 运行失败 | 2.3 PoCL clEnqueueCopyBuffer 路径 NULL 调用 |
| `image_copy` | 能力缺失 | 2.3 PoCL 报 no image support |
| `image_hwsw` | 能力缺失 | 2.3 无 image 支持，且无 TEX/RTU 单元 |
| `image_int` | 能力缺失 | 2.3 PoCL 报 no image support |
| `parboil-histo` | 工具链限制 | 2.3 PoCL 以 OpenCL C 1.2 编译，内核原子内建缺少 `_cl_atomic_add/min/max` |
| `parboil-mri-gridding` | 工具链限制 | 同上，缺少 `_cl_atomic_add/_sub`（链接失败） |
| `shoc-sort` | 模型不兼容 | 2.3 simx 卡死；rtlsim 调度器 stall timeout（死锁） |
| `ventus-softmax` | 模型不兼容 | 2.3 simx 结果错误（归一化因子=最大指数项）并触发 simobject 断言；rtlsim PASS |

## 失败原因分析（能力缺失/原子之外）

| case | simx | rtlsim | 证据与结论 | 置信度 |
|---|---|---|---|---|
| `blueporcelain-nqueen` | divergent branch! PC=0x80000194 并 abort | FAILED! - 1（期望 92，实际 82） | 2.3 simx execute.cpp:446 要求同一 warp 活跃 lane 的分支结果一致；保留的内核 ELF 反汇编显示该 PC 位于回溯 while(i>=0) 的条件分支（0x80000190 bltz，循环体 0x80000194），每个 work-item 回溯深度不同，必然发散。 2.3 simx 的发散限制 + RTL 发散执行错误；测试本身在 3.0 双驱动通过。 | 高（simx 已定位到源码/反汇编；RTL 具体缺陷未定位） |
| `btree` | 卡死（>150s） | VX_schedule.sv:415 stall timeout 断言（cycle 414351） | 同 shoc-sort；barrier 位于统一 for(height) 循环，work-group=ORDER=16，无发散 barrier。 2.3 调度/RTL 死锁；3.0 双驱动通过。 | 中（同 shoc-sort） |
| `cfd` | Segmentation fault | Segmentation fault | gdb 栈：NULL 调用，pocl_exec_command -> vortex_command_scheduler -> pocl_vortex_submit -> clEnqueueCopyBuffer -> main。 2.3 PoCL Vortex 设备的 CopyBuffer 路径 NULL 回调；测试只调用标准 OpenCL API。 | 高（gdb 栈） |
| `copybuf` | Segmentation fault | Segmentation fault | gdb 栈与 cfd 相同（test_basic_copy -> clEnqueueCopyBuffer -> ... NULL）。 同一 PoCL/runtime CopyBuffer bug；测试有效。 | 高（gdb 栈） |
| `shoc-sort` | 卡死（>150s，100% CPU） | VX_schedule.sv:415 stall timeout 断言（cycle 888165） | RTL 断言为“所有 warp stalled 超时”；NUM_BARRIERS 由 2 改为 8 重建后断言周期完全相同，排除 barrier ID 不足；内核 barrier 均位于统一循环（window += local_size），无发散 barrier。 2.3 调度/内存/同步路径的死锁（模型/RTL bug）；3.0 双驱动通过。 | 中（死锁确认；内部根因未定位） |
| `ventus-softmax` | rows 结果错误，随后 chw 触发 simobject 端口断言 | PASS（64/32 位） | N=8/16/32/64 下 rows 全部错误，且实际值恰为 e_j/e_max（实际最大值=1.0，期望最大值 0.582595），即归一化因子被当作最大指数项；同一 ELF 在 rtlsim 上正确。 2.3 simx 执行/FP 语义 bug；测试有效。 | 高（rtlsim 对照 + 数值规律；simx 内部指令未定位） |

## 结论与边界

- TESTS 的 84 个用例名称与顺序与 3.0 的 tests/opencl-unique 完全一致。
- 默认扫描 71 个；rtlsim 在 64/32 位均 71/71 PASS。
- simx 在完整扫描中各有 1 次间歇中止（64: lstm，32: gru），单独重复 3/3 PASS。
- 2.3 的 simx/rtlsim instrs 不要求相等；结果中保留两侧数值供参考。
- 已重建 EXT_A runtime 并补跑 4 个原子用例；tpacf 通过，其余三个因 OpenCL 1.2 内核库缺少 _cl_atomic_* 而构建失败。
- image_* 在 2.3 上明确不受支持，归入本版本的问题清单。
- 失败原因分析：nqueen 为 2.3 simx 发散限制 + RTL 发散执行错误；sort/btree 为 2.3 调度死锁；softmax 为 2.3 simx 执行语义 bug（rtlsim 正确）；cfd/copybuf 为 PoCL CopyBuffer NULL 调用。详见 diagnosis。

原始数据：`results/v2.3.json`；每个用例的完整日志未入库。
