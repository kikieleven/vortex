# Vortex 2.2 — opencl-unique 测试结果

- 版本：Vortex 2.2，commit `e86eeab8e`（分支 `vortex2.2`）
- 日期：2026-11-25
- 构建：`build/`（XLEN=64）、`build32/`（XLEN=32）；驱动：simx、rtlsim
- 用例：TESTS 84，默认扫描 69，EXCLUDE 15
- 用例名与顺序与 3.0 的 `tests/opencl-unique` **完全一致**（84/84）

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

## 失败 / 未运行用例及原因

| case | 原因 |
|---|---|
| `atomicreduce` | NOT RUN：需要专用的 EXT_A runtime 重建 |
| `blueporcelain-nqueen` | FAIL：2.2 报 "divergent branch" 并中止 |
| `btree` | TIMEOUT：2.2 卡死 |
| `cfd` | FAIL：host 段错误 |
| `copybuf` | FAIL：host 段错误 |
| `image_copy` | FAIL：2.2 PoCL 无 image 支持 |
| `image_hwsw` | FAIL：2.2 PoCL 无 image 支持，且无 TEX/RTU |
| `image_int` | FAIL：2.2 PoCL 无 image 支持 |
| `nw` | FAIL：32 位结果错误（64 位 PASS） |
| `parboil-histo` | NOT RUN：需要专用的 EXT_A runtime 重建 |
| `parboil-mri-gridding` | NOT RUN：需要专用的 EXT_A runtime 重建 |
| `parboil-tpacf` | NOT RUN：需要专用的 EXT_A runtime 重建（2.2 common.mk 不传 CONFIGS） |
| `rodinia-srad-v2` | FAIL：64 位结果错误（32 位 PASS） |
| `shoc-sort` | TIMEOUT：2.2 卡死 |
| `ventus-softmax` | FAIL：结果错误（64 位还会卡死） |

## 结论与边界

- TESTS 的 84 个用例名称与顺序与 3.0 的 tests/opencl-unique 完全一致。
- 最终验证按 64 simx -> 64 rtlsim -> 32 simx -> 32 rtlsim 串行执行：2.2 的 kernel/scripts/vxbin.py 把临时文件硬编码为 /tmp/temp_kernel.bin，两个 build 树并发会互相覆盖，产生假失败（vectormma/shoc-triad/triangle-area 已确认）。
- 默认扫描 69 个；详细逐用例结果见 RESULTS.md。
- 原子用例未运行；2.2 需要专用的 EXT_A runtime 重建。
- 2.2 的 simx/rtlsim instrs 不要求相等；结果中保留两侧数值供参考。

原始数据：`results/v2.2.json`；每个用例的完整日志未入库。
