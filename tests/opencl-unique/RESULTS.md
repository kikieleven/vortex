# Vortex 3.0 — opencl-unique 测试结果

- 版本：Vortex 3.0，commit `3460fa819`
- 日期：2026-11-26
- 构建：`build/`（XLEN=64）、`build32/`（XLEN=32）；驱动：simx、rtlsim
- 用例：TESTS 84，默认扫描 76，EXCLUDE 8
- 用例名与顺序与 3.0 的 `tests/opencl-unique` 完全一致（84/84）

## 默认扫描结果

| case | 64 simx | 64 rtlsim | 32 simx | 32 rtlsim | instrs 64 (s/r) | instrs 32 (s/r) |
|---|---|---|---|---|---|---|
| `backprop` | PASS | PASS | PASS | PASS | 6228/6228 | 6118/6118 |
| `blackscholes` | PASS | PASS | PASS | PASS | 106375/106375 | 822735/822735 |
| `blueporcelain-bitonic-sort` | PASS | PASS | PASS | PASS | 31730/31730 | 30142/30142 |
| `blueporcelain-dot-product` | PASS | PASS | PASS | PASS | 32960/32960 | 29836/29836 |
| `blueporcelain-matrix-inversion` | PASS | PASS | PASS | PASS | 2449088/2449088 | 2454976/2454976 |
| `blueporcelain-matrix-transpose` | PASS | PASS | PASS | PASS | 238048/238048 | 225740/225740 |
| `blueporcelain-nbody` | PASS | PASS | PASS | PASS | 79072/79072 | 74700/74700 |
| `blueporcelain-nn-op-adaptiveavgpool2d` | PASS | PASS | PASS | PASS | 553/553 | 534/534 |
| `blueporcelain-nn-op-batchnorm3d` | PASS | PASS | PASS | PASS | 1332/1332 | 1305/1305 |
| `blueporcelain-nn-op-conv3d` | PASS | PASS | PASS | PASS | 5928/5928 | 5712/5712 |
| `blueporcelain-nn-op-dropout` | PASS | PASS | PASS | PASS | 1786/1786 | 1766/1766 |
| `blueporcelain-nn-op-embedding` | PASS | PASS | PASS | PASS | 2528/2528 | 2444/2444 |
| `blueporcelain-nn-op-gelu` | PASS | PASS | PASS | PASS | 3426/3426 | 3406/3406 |
| `blueporcelain-nn-op-gru` | PASS | PASS | PASS | PASS | 6559/6559 | 6497/6497 |
| `blueporcelain-nn-op-hardsigmoid` | PASS | PASS | PASS | PASS | 1408/1408 | 1388/1388 |
| `blueporcelain-nn-op-hardswish` | PASS | PASS | PASS | PASS | 1424/1424 | 1404/1404 |
| `blueporcelain-nn-op-layernorm` | PASS | PASS | PASS | PASS | 368/368 | 363/363 |
| `blueporcelain-nn-op-leakyrelu` | PASS | PASS | PASS | PASS | 1452/1452 | 1432/1432 |
| `blueporcelain-nn-op-linear` | PASS | PASS | PASS | PASS | 980/980 | 944/944 |
| `blueporcelain-nn-op-lstm` | PASS | PASS | PASS | PASS | 9624/9624 | 9550/9550 |
| `blueporcelain-nn-op-sigmoid` | PASS | PASS | PASS | PASS | 2792/2792 | 2772/2772 |
| `blueporcelain-nn-op-silu` | PASS | PASS | PASS | PASS | 2792/2792 | 2772/2772 |
| `blueporcelain-nn-op-tanh` | PASS | PASS | PASS | PASS | 2776/2776 | 2756/2756 |
| `blueporcelain-nqueen` | PASS | PASS | PASS | PASS | 37797/37797 | 37461/37461 |
| `blueporcelain-reduction-max` | PASS | PASS | PASS | PASS | 48506/48506 | 46694/46694 |
| `blueporcelain-triangle-area` | PASS | PASS | PASS | PASS | 5632/5632 | 5612/5612 |
| `blueporcelain-vectoradd` | PASS | PASS | PASS | PASS | 11616/11616 | 11940/11940 |
| `blueporcelain-vectordiv` | PASS | PASS | PASS | PASS | 3872/3872 | 3852/3852 |
| `blueporcelain-vectormma` | PASS | PASS | PASS | PASS | 7776/7776 | 7756/7756 |
| `btree` | PASS | PASS | PASS | PASS | 44110/44110 | 46780/46780 |
| `cfd` | PASS | PASS | PASS | PASS | 52166/52166 | 49732/49732 |
| `dwt2d` | PASS | PASS | PASS | PASS | 4020/4020 | 3768/3768 |
| `heartwall` | PASS | PASS | PASS | PASS | 62231/62231 | 56080/56080 |
| `hotspot3D` | PASS | PASS | PASS | PASS | 30656/30656 | 30104/30104 |
| `kmeans` | PASS | PASS | PASS | PASS | -/- | -/- |
| `myocyte` | PASS | PASS | PASS | PASS | 320395/320395 | 4719501/4719501 |
| `nearn` | PASS | PASS | PASS | PASS | 1243304/1243304 | 1243284/1243284 |
| `nw` | PASS | PASS | PASS | PASS | 26882/26882 | 23584/23584 |
| `oclprintf` | PASS | PASS | PASS | PASS | 6098/6098 | 5853/5853 |
| `parboil-lbm` | PASS | PASS | PASS | PASS | 127344/127344 | 126132/126132 |
| `parboil-mri-q` | PASS | PASS | PASS | PASS | 406816/406816 | 17392717/17392717 |
| `parboil-sad` | PASS | PASS | PASS | PASS | 4961284/4961284 | 4662552/4662552 |
| `parboil-spmv` | PASS | PASS | PASS | PASS | 9120/9120 | 9100/9100 |
| `polybench-3mm` | PASS | PASS | PASS | PASS | 229536/229536 | 228708/228708 |
| `polybench-adi` | PASS | PASS | PASS | PASS | 53784/53784 | 51984/51984 |
| `polybench-covariance` | PASS | PASS | PASS | PASS | 332480/332480 | 332388/332388 |
| `polybench-doitgen` | PASS | PASS | PASS | PASS | 111312/111312 | 110400/110400 |
| `polybench-gemver` | PASS | PASS | PASS | PASS | 100576/100576 | 100484/100484 |
| `polybench-lu` | PASS | PASS | PASS | PASS | 87076/87076 | 86436/86436 |
| `polybench-syr2k` | PASS | PASS | PASS | PASS | 134880/134880 | 134860/134860 |
| `rodinia-leukocyte` | PASS | PASS | PASS | PASS | 10854053/10854053 | 10900780/10900780 |
| `rodinia-particlefilter-single` | PASS | PASS | PASS | PASS | 50067/50067 | 309367/309367 |
| `rodinia-srad-v2` | PASS | PASS | PASS | PASS | 231084/231084 | 228988/228988 |
| `saxpy` | PASS | PASS | PASS | PASS | 52704/52704 | 48588/48588 |
| `shoc-bfs` | PASS | PASS | PASS | PASS | 307742/307742 | 310186/310186 |
| `shoc-md` | PASS | PASS | PASS | PASS | 960626/960626 | 862558/862558 |
| `shoc-md5hash` | PASS | PASS | PASS | PASS | 723000/723000 | 722596/722596 |
| `shoc-reduction` | PASS | PASS | PASS | PASS | 60192/60192 | 57100/57100 |
| `shoc-scan` | PASS | PASS | PASS | PASS | 52364/52364 | 49380/49380 |
| `shoc-sort` | PASS | PASS | PASS | PASS | 4200384/4200384 | 3999424/3999424 |
| `shoc-spmv` | PASS | PASS | PASS | PASS | 8084/8084 | 7808/7808 |
| `shoc-stencil2d` | PASS | PASS | PASS | PASS | 190144/190144 | 161304/161304 |
| `shoc-triad` | PASS | PASS | PASS | PASS | 55776/55776 | 55756/55756 |
| `spmv` | PASS | PASS | PASS | PASS | 152684/152684 | 152664/152664 |
| `srad` | PASS | PASS | PASS | PASS | 201820/201820 | 1000248/1000248 |
| `ventus-batchnorm2d` | PASS | PASS | PASS | PASS | 10752/10752 | 10732/10732 |
| `ventus-conv` | PASS | PASS | PASS | PASS | 439094/439094 | 389666/389666 |
| `ventus-conv-bn` | PASS | PASS | PASS | PASS | 448480/448480 | 399052/399052 |
| `ventus-conv-fused` | PASS | PASS | PASS | PASS | 413664/413664 | 385100/385100 |
| `ventus-gemm` | PASS | PASS | PASS | PASS | 49888/49888 | 50380/50380 |
| `ventus-gemm-fused` | PASS | PASS | PASS | PASS | 52322/52322 | 52302/52302 |
| `ventus-mnist` | PASS | PASS | PASS | PASS | 156295/156295 | 156255/156255 |
| `ventus-mnist-conv-small` | PASS | PASS | PASS | PASS | 188007/188007 | 167085/167085 |
| `ventus-pool2d` | PASS | PASS | PASS | PASS | 10980/10980 | 10472/10472 |
| `ventus-relu` | PASS | PASS | PASS | PASS | 8832/8832 | 8812/8812 |
| `ventus-softmax` | PASS | PASS | PASS | PASS | 25972/25972 | 25948/25948 |

## 特殊配置用例（已验证通过，不属于问题）

| case | 说明 | 64 simx | 64 rtlsim | 32 simx | 32 rtlsim |
|---|---|---|---|---|---|
| `atomicreduce` | 运行需 CONFIGS=-DVX_CFG_EXT_A_ENABLE | PASS | PASS | PASS | PASS |
| `copybuf` | 只测试 buffer copy API，不启动 kernel（instrs=0） | PASS | PASS | PASS | PASS |
| `image_copy` | 需要 image-enabled PoCL（3.0 支持） | PASS | PASS | PASS | PASS |
| `image_hwsw` | 需要 image-enabled PoCL 与 TEX/RTU（3.0 支持） | PASS | PASS | PASS | PASS |
| `image_int` | 需要 image-enabled PoCL（3.0 支持） | PASS | PASS | PASS | PASS |
| `parboil-histo` | 运行需 CONFIGS=-DVX_CFG_EXT_A_ENABLE | PASS | PASS | PASS | PASS |
| `parboil-mri-gridding` | 运行需 CONFIGS=-DVX_CFG_EXT_A_ENABLE | PASS | PASS | PASS | PASS |
| `parboil-tpacf` | 运行需 EXT_A（实测默认也能通过，移植不含原子内建） | PASS | PASS | PASS | PASS |

## 本版本有问题的测试

无。本版本没有被判定为有问题的测试。

## 结论与边界

- TESTS 的 84 个用例名称与顺序与 2.2/2.3 移植版本完全一致。
- 默认扫描 76 个；64/32 位的 simx 与 rtlsim 均 PASS。
- 8 个 EXCLUDE 用例已手工补跑：原子类带 CONFIGS=-DVX_CFG_EXT_A_ENABLE，image/copybuf 直接用默认配置，均 PASS。
- 3.0 的 simx/rtlsim instrs 在确定性用例上完全一致（原子用例除外；本套 EXCLUDE 中 copybuf 的 instrs=0）。
- 本版本没有被判定为有问题的测试；8 个 EXCLUDE 均为特殊配置且已全部通过。
- 本版本没有被判定为有问题的测试；8 个 EXCLUDE 均为特殊配置且已全部通过，因此没有失败原因分析。

原始数据：`results/v3.0.json`；每个用例的完整日志未入库。
