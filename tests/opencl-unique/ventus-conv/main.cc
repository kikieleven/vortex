// conv2d (Ventus-OpenCL-Testcase, _get_case/AIops/Conv) — standalone
// self-checking OpenCL port for Vortex.
//
//   out[oc,oy,ox] = bias[oc] + sum_{ic,ky,kx} in[ic,iy,ix] * w[oc,ic,ky,kx]
//                   (optionally ReLU'd)
//   iy = oy*stride_h - pad_h + ky,  ix = ox*stride_w - pad_w + kx
//
// Upstream: third_benchmark/Ventus-OpenCL-Testcase/_get_case/AIops/Conv/
// (kernel.cl is the upstream conv.cl; the CPU reference below reproduces the
// kernel's loop nesting and accumulation order term by term).
//
// Differences from the upstream host:
//   * the launch keeps the upstream 3D NDRange {out_c, out_h, out_w}, but the
//     global size is rounded up to a multiple of the work-group. Vortex caps a
//     work-group at 16 work-items, so the local size is {1,4,4}; the kernel's
//     `oc/oy/ox` guard consumes the extra work-items.
//   * inputs are generated with a private deterministic LCG instead of rand():
//     PoCL shares the process-wide rand() state, which makes the simx/rtlsim
//     retired-instruction counts differ between runs (see the porting guide
//     §13.2).
//   * comparison is |actual - expected| <= REL_TOL*|expected| + ABS_TOL, the
//     mixed form used by tests/opencl/{hotspot3D,cfd,lud}. ULP is not usable
//     here: the stencil sums signed products and the result can cross zero.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "conv2d"

#define REL_TOL 1e-3f
#define ABS_TOL 1e-3f

#define CL_CHECK(_expr)                                                \
   do {                                                                \
     cl_int _err = _expr;                                              \
     if (_err == CL_SUCCESS)                                           \
       break;                                                          \
     printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);   \
     cleanup();                                                        \
     exit(-1);                                                         \
   } while (0)

#define CL_CHECK2(_expr)                                               \
   ({                                                                  \
     cl_int _err = CL_INVALID_VALUE;                                   \
     decltype(_expr) _ret = _expr;                                     \
     if (_err != CL_SUCCESS) {                                         \
       printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
       cleanup();                                                      \
       exit(-1);                                                       \
     }                                                                 \
     _ret;                                                             \
   })

// Deterministic LCG: the same values on every run, independent of anything the
// OpenCL runtime does with rand().
static uint32_t lcg_state = 1u;
static float generate() {                     // [0, 1)
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<float>(lcg_state >> 8) / static_cast<float>(1u << 24);
}
static float generate_pm1() { return generate() * 2.0f - 1.0f; }  // [-1, 1)

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    return -1;
  }

  fseek(fp, 0, SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return 0;
}

// Same loop nesting, same accumulation order, same boundary skips as kernel.cl.
static void conv2d_cpu(float* out, const float* in, const float* w, const float* b,
                       int in_c, int in_h, int in_w,
                       int out_c, int k_h, int k_w,
                       int out_h, int out_w,
                       int stride_h, int stride_w, int pad_h, int pad_w, int do_relu) {
  const int inHW = in_h * in_w;
  const int outHW = out_h * out_w;
  for (int oc = 0; oc < out_c; ++oc) {
    for (int oy = 0; oy < out_h; ++oy) {
      for (int ox = 0; ox < out_w; ++ox) {
        float sum = b[oc];
        for (int ic = 0; ic < in_c; ++ic) {
          for (int ky = 0; ky < k_h; ++ky) {
            const int iy = oy * stride_h - pad_h + ky;
            if (iy < 0 || iy >= in_h) continue;
            for (int kx = 0; kx < k_w; ++kx) {
              const int ix = ox * stride_w - pad_w + kx;
              if (ix < 0 || ix >= in_w) continue;
              const int in_idx = ic * inHW + iy * in_w + ix;
              const int w_idx = oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + ky * k_w + kx;
              sum += in[in_idx] * w[w_idx];
            }
          }
        }
        if (do_relu && sum < 0.0f) sum = 0.0f;
        out[oc * outHW + oy * out_w + ox] = sum;
      }
    }
  }
}

static size_t round_up(size_t value, size_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem in_memobj = NULL;
cl_mem w_memobj = NULL;
cl_mem b_memobj = NULL;
cl_mem o_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (in_memobj) clReleaseMemObject(in_memobj);
  if (w_memobj) clReleaseMemObject(w_memobj);
  if (b_memobj) clReleaseMemObject(b_memobj);
  if (o_memobj) clReleaseMemObject(o_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

// Defaults follow the upstream host: 3x16x16 input, 8 output channels, 3x3
// kernel, stride 1, padding 1, ReLU on.
int size = 16;
int do_relu = 1;

static void show_usage() {
  printf("Usage: [-n size] [-r do_relu(0|1)] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:r:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
      break;
    case 'r':
      do_relu = atoi(optarg);
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }

  printf("float workload size=%d do_relu=%d\n", size, do_relu);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 1) {
    printf("Error: size must be at least 1\n");
    return -1;
  }

  const int in_c = 3;
  const int in_h = size;
  const int in_w = size;
  const int out_c = 8;
  const int k_h = 3;
  const int k_w = 3;
  const int stride_h = 1;
  const int stride_w = 1;
  const int pad_h = 1;
  const int pad_w = 1;

  const int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
  const int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
  if (out_h <= 0 || out_w <= 0) {
    printf("Error: invalid output size out_h=%d out_w=%d\n", out_h, out_w);
    return -1;
  }

  const size_t in_size = (size_t)in_c * in_h * in_w;
  const size_t w_size = (size_t)out_c * in_c * k_h * k_w;
  const size_t b_size = (size_t)out_c;
  const size_t o_size = (size_t)out_c * out_h * out_w;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  in_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, in_size * sizeof(float), NULL, &_err));
  w_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, w_size * sizeof(float), NULL, &_err));
  b_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, b_size * sizeof(float), NULL, &_err));
  o_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, o_size * sizeof(float), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  int arg = 0;
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&in_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&w_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&b_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&o_memobj));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&in_c));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&in_h));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&in_w));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&out_c));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&k_h));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&k_w));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&out_h));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&out_w));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&stride_h));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&stride_w));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&pad_h));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&pad_w));
  CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&do_relu));

  // Host-side input/output buffers
  std::vector<float> h_in(in_size);
  std::vector<float> h_w(w_size);
  std::vector<float> h_b(b_size);
  std::vector<float> h_o(o_size, 0.0f);

  for (size_t i = 0; i < in_size; ++i)
    h_in[i] = generate_pm1();
  for (size_t i = 0; i < w_size; ++i)
    h_w[i] = generate_pm1();
  for (size_t i = 0; i < b_size; ++i)
    h_b[i] = generate_pm1();

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, in_memobj, CL_TRUE, 0, in_size * sizeof(float), h_in.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, w_memobj, CL_TRUE, 0, w_size * sizeof(float), h_w.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_memobj, CL_TRUE, 0, b_size * sizeof(float), h_b.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  // Upstream 3D NDRange; the global size is rounded up to the work-group size
  // because Vortex caps a work-group at 16 work-items.
  size_t local_work_size[3] = {1, 4, 4};
  size_t global_work_size[3];
  global_work_size[0] = round_up((size_t)out_c, local_work_size[0]);
  global_work_size[1] = round_up((size_t)out_h, local_work_size[1]);
  global_work_size[2] = round_up((size_t)out_w, local_work_size[2]);
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, o_memobj, CL_TRUE, 0, o_size * sizeof(float), h_o.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<float> h_ref(o_size);
  conv2d_cpu(h_ref.data(), h_in.data(), h_w.data(), h_b.data(),
             in_c, in_h, in_w, out_c, k_h, k_w, out_h, out_w,
             stride_h, stride_w, pad_h, pad_w, do_relu);
  int errors = 0;
  for (size_t i = 0; i < o_size; ++i) {
    float a = h_o[i];
    float b = h_ref[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, b, a, fabsf(a - b));
      }
      ++errors;
    }
  }
  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors;
}