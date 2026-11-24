// MNIST fc forward (Ventus-OpenCL-Testcase, _get_case/MNIST) — standalone
// self-checking OpenCL port for Vortex.
//
// Two fc_layer launches, exactly like the upstream host:
//   hidden = relu(W1 * input + bias1)      784 -> 128, activation = 1
//   output =       W2 * hidden + bias2     128 ->  10, activation = 0
//
// Upstream: third_benchmark/Ventus-OpenCL-Testcase/_get_case/MNIST/ (kernel.cl
// is the upstream nn_forward.cl). The upstream host loads pre-computed weights
// and expected outputs from `data_gen/*.txt`; this port instead generates its
// own data with a deterministic LCG and computes the reference on the CPU, so
// the repository does not carry 16 MB of upstream data. The layer shapes stay
// the upstream ones and are command-line adjustable.
//
// Differences from the upstream host:
//   * inputs/weights/biases come from a private deterministic LCG in [-1,1)
//     instead of data files (porting guide §13.2). Signed values make the
//     ReLU branch of the first layer actually fire.
//   * both layers are verified: the hidden vector is downloaded and compared
//     too, not only the final 10 outputs (upstream only checks the last layer).
//   * each launch is 1D with local size 16; the global size is rounded up and
//     the kernel's `idx < output_size` guard drops the extra work-items.
//   * comparison is |actual - expected| <= REL_TOL*|expected| + ABS_TOL: the
//     dot products are 784/128-term accumulations and the first layer is
//     ReLU'd, so half the hidden values are exactly 0 and the rest are signed
//     (porting guide §4.7).

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "fc_layer"

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

// Deterministic LCG (porting guide §13.2).
static uint32_t lcg_state = 1u;
static float generate_pm1() {                 // [-1, 1)
  lcg_state = lcg_state * 1664525u + 1013904223u;
  return static_cast<float>(lcg_state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
}

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

// Same accumulation order, bias addition and activation as kernel.cl.
static void fc_layer_cpu(float* out, const float* in, const float* w,
                         const float* b, int input_size, int output_size,
                         int activation) {
  for (int idx = 0; idx < output_size; ++idx) {
    float sum = 0.0f;
    for (int i = 0; i < input_size; ++i) {
      sum += in[i] * w[idx * input_size + i];
    }
    sum += b[idx];
    if (activation == 1) {
      sum = fmaxf(sum, 0.0f);
    }
    out[idx] = sum;
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
cl_mem w1_memobj = NULL;
cl_mem b1_memobj = NULL;
cl_mem h_memobj = NULL;
cl_mem w2_memobj = NULL;
cl_mem b2_memobj = NULL;
cl_mem o_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (in_memobj) clReleaseMemObject(in_memobj);
  if (w1_memobj) clReleaseMemObject(w1_memobj);
  if (b1_memobj) clReleaseMemObject(b1_memobj);
  if (h_memobj) clReleaseMemObject(h_memobj);
  if (w2_memobj) clReleaseMemObject(w2_memobj);
  if (b2_memobj) clReleaseMemObject(b2_memobj);
  if (o_memobj) clReleaseMemObject(o_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int input_size = 784;
int hidden_size = 128;
int output_size = 10;

static void show_usage() {
  printf("Usage: [-i input_size] [-m hidden_size] [-o output_size] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "i:m:o:h")) != -1) {
    switch (c) {
    case 'i':
      input_size = atoi(optarg);
      break;
    case 'm':
      hidden_size = atoi(optarg);
      break;
    case 'o':
      output_size = atoi(optarg);
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

  printf("float workload input=%d hidden=%d output=%d\n",
         input_size, hidden_size, output_size);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (input_size < 1 || hidden_size < 1 || output_size < 1) {
    printf("Error: all layer sizes must be at least 1\n");
    return -1;
  }

  const int I = input_size;
  const int H = hidden_size;
  const int O = output_size;

  const size_t in_n = (size_t)I;
  const size_t w1_n = (size_t)H * I;
  const size_t b1_n = (size_t)H;
  const size_t h_n  = (size_t)H;
  const size_t w2_n = (size_t)O * H;
  const size_t b2_n = (size_t)O;
  const size_t o_n  = (size_t)O;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  in_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, in_n * sizeof(float), NULL, &_err));
  w1_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, w1_n * sizeof(float), NULL, &_err));
  b1_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, b1_n * sizeof(float), NULL, &_err));
  h_memobj  = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, h_n * sizeof(float), NULL, &_err));
  w2_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, w2_n * sizeof(float), NULL, &_err));
  b2_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, b2_n * sizeof(float), NULL, &_err));
  o_memobj  = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, o_n * sizeof(float), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  std::vector<float> h_in(in_n), h_w1(w1_n), h_b1(b1_n);
  std::vector<float> h_w2(w2_n), h_b2(b2_n);
  std::vector<float> h_h(h_n, 0.0f), h_o(o_n, 0.0f);
  for (size_t i = 0; i < in_n; ++i) h_in[i] = generate_pm1();
  for (size_t i = 0; i < w1_n; ++i) h_w1[i] = generate_pm1();
  for (size_t i = 0; i < b1_n; ++i) h_b1[i] = generate_pm1();
  for (size_t i = 0; i < w2_n; ++i) h_w2[i] = generate_pm1();
  for (size_t i = 0; i < b2_n; ++i) h_b2[i] = generate_pm1();

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, in_memobj, CL_TRUE, 0, in_n * sizeof(float), h_in.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, w1_memobj, CL_TRUE, 0, w1_n * sizeof(float), h_w1.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b1_memobj, CL_TRUE, 0, b1_n * sizeof(float), h_b1.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, w2_memobj, CL_TRUE, 0, w2_n * sizeof(float), h_w2.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b2_memobj, CL_TRUE, 0, b2_n * sizeof(float), h_b2.data(), 0, NULL, NULL));

  size_t local_work_size[1] = {16};

  // ---- layer 1: hidden = relu(W1 * input + bias1) --------------------
  printf("Execute the kernel (layer 1: %d -> %d, relu)\n", I, H);
  {
    int arg = 0;
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&I));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&H));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&in_memobj));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&w1_memobj));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&b1_memobj));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&h_memobj));
    int activation = 1;
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&activation));

    size_t global_work_size[1] = {round_up((size_t)H, local_work_size[0])};
    auto time_start = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);
  }

  printf("Download destination buffer (layer 1)\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, h_memobj, CL_TRUE, 0, h_n * sizeof(float), h_h.data(), 0, NULL, NULL));

  // CPU reference for layer 1 (the output feeds layer 2's reference).
  std::vector<float> h_h_ref(h_n);
  fc_layer_cpu(h_h_ref.data(), h_in.data(), h_w1.data(), h_b1.data(), I, H, 1);

  // ---- layer 2: output = W2 * hidden + bias2 -------------------------
  printf("Execute the kernel (layer 2: %d -> %d, no activation)\n", H, O);
  {
    int arg = 0;
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&H));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&O));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&h_memobj));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&w2_memobj));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&b2_memobj));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&o_memobj));
    int activation = 0;
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&activation));

    size_t global_work_size[1] = {round_up((size_t)O, local_work_size[0])};
    auto time_start = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);
  }

  printf("Download destination buffer (layer 2)\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, o_memobj, CL_TRUE, 0, o_n * sizeof(float), h_o.data(), 0, NULL, NULL));

  std::vector<float> h_o_ref(o_n);
  fc_layer_cpu(h_o_ref.data(), h_h_ref.data(), h_w2.data(), h_b2.data(), H, O, 0);

  printf("Verify result\n");
  int errors = 0;
  for (size_t i = 0; i < h_n; ++i) {
    float a = h_h[i];
    float b = h_h_ref[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: layer1[%d] expected=%f, actual=%f (diff=%g)\n",
               (int)i, b, a, fabsf(a - b));
      }
      ++errors;
    }
  }
  for (size_t i = 0; i < h_o.size(); ++i) {
    float a = h_o[i];
    float b = h_o_ref[i];
    float tol = REL_TOL * fabsf(b) + ABS_TOL;
    if (fabsf(a - b) > tol) {
      if (errors < 100) {
        printf("*** error: layer2[%d] expected=%f, actual=%f (diff=%g)\n",
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