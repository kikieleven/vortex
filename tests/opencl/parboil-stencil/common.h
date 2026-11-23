#ifndef COMMON_H
#define COMMON_H

/* Same flattening macro as Parboil's src/opencl_base/common.h. */
#define Index3D(_nx,_ny,_i,_j,_k) ((_i)+_nx*((_j)+_ny*(_k)))

#endif