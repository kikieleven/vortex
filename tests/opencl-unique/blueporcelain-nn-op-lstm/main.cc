#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <vector>
#include <chrono>
#include "common.h"

// Deterministic LCG in place of the process-wide rand(): PoCL also consumes
// rand(), so separate simx/rtlsim processes would otherwise get different
// input data and their instruction counts could not be compared.
static uint32_t g_lcg_state = 1u;
static uint32_t lcg_rand() {
  g_lcg_state = g_lcg_state * 1664525u + 1013904223u;
  return g_lcg_state >> 1;  // 31 bits: same range as RAND_MAX
}


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

static int read_kernel_file(const char* filename, uint8_t** data, size_t* size) {
  if (nullptr == filename || nullptr == data || 0 == size)
    return -1;

  FILE* fp = fopen(filename, "r");
  if (NULL == fp) {
    fprintf(stderr, "Failed to load kernel.");
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

static float sigmoidf_(float x) {
  return 1.0f / (1.0f + expf(-x));
}

/* One step of the PyTorch LSTM recurrence; gates packed i,f,g,o. */
static void lstm_step_cpu(float* h_next, float* c_next, const float* x,
                          const float* h_prev, const float* c_prev,
                          const float* w_ih, const float* w_hh,
                          const float* b_ih, const float* b_hh,
                          int in_features, int hidden) {
  for (int j = 0; j < hidden; ++j) {
    float gi_i = b_ih[0 * hidden + j], gh_i = b_hh[0 * hidden + j];
    float gi_f = b_ih[1 * hidden + j], gh_f = b_hh[1 * hidden + j];
    float gi_g = b_ih[2 * hidden + j], gh_g = b_hh[2 * hidden + j];
    float gi_o = b_ih[3 * hidden + j], gh_o = b_hh[3 * hidden + j];

    for (int k = 0; k < in_features; ++k) {
      float xv = x[k];
      gi_i += w_ih[(0 * hidden + j) * in_features + k] * xv;
      gi_f += w_ih[(1 * hidden + j) * in_features + k] * xv;
      gi_g += w_ih[(2 * hidden + j) * in_features + k] * xv;
      gi_o += w_ih[(3 * hidden + j) * in_features + k] * xv;
    }
    for (int k = 0; k < hidden; ++k) {
      float hv = h_prev[k];
      gh_i += w_hh[(0 * hidden + j) * hidden + k] * hv;
      gh_f += w_hh[(1 * hidden + j) * hidden + k] * hv;
      gh_g += w_hh[(2 * hidden + j) * hidden + k] * hv;
      gh_o += w_hh[(3 * hidden + j) * hidden + k] * hv;
    }

    float i_gate = sigmoidf_(gi_i + gh_i);
    float f_gate = sigmoidf_(gi_f + gh_f);
    float g_gate = tanhf(gi_g + gh_g);
    float o_gate = sigmoidf_(gi_o + gh_o);

    float c = f_gate * c_prev[j] + i_gate * g_gate;
    c_next[j] = c;
    h_next[j] = o_gate * tanhf(c);
  }
}

static cl_device_id device_id = NULL;
static cl_context context = NULL;
static cl_command_queue commandQueue = NULL;
static cl_program program = NULL;
static cl_kernel kernel = NULL;
static cl_mem x_memobj = NULL;
static cl_mem h_memobj[2] = {NULL, NULL};
static cl_mem c_memobj[2] = {NULL, NULL};
static cl_mem w_ih_memobj = NULL;
static cl_mem w_hh_memobj = NULL;
static cl_mem b_ih_memobj = NULL;
static cl_mem b_hh_memobj = NULL;
static uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (x_memobj) clReleaseMemObject(x_memobj);
  if (h_memobj[0]) clReleaseMemObject(h_memobj[0]);
  if (h_memobj[1]) clReleaseMemObject(h_memobj[1]);
  if (c_memobj[0]) clReleaseMemObject(c_memobj[0]);
  if (c_memobj[1]) clReleaseMemObject(c_memobj[1]);
  if (w_ih_memobj) clReleaseMemObject(w_ih_memobj);
  if (w_hh_memobj) clReleaseMemObject(w_hh_memobj);
  if (b_ih_memobj) clReleaseMemObject(b_ih_memobj);
  if (b_hh_memobj) clReleaseMemObject(b_hh_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

static int local_size = 4;

static void show_usage() {
  printf("Usage: [-l local_size] [-h: help]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "l:h")) != -1) {
    switch (c) {
    case 'l':
      local_size = atoi(optarg);
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

  if (local_size <= 0) {
    printf("Error: local_size must be positive\n");
    exit(-1);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  cl_platform_id platform_id;
  size_t kernel_size;

  const int seq = LSTM_SEQ, in_features = LSTM_IN, hidden = LSTM_HIDDEN;

  printf("Workload seq=%d in=%d hidden=%d, local_size=%d\n",
         seq, in_features, hidden, local_size);

  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  size_t gate_h = (size_t)LSTM_GATES * hidden;
  size_t w_ih_count = gate_h * in_features;
  size_t w_hh_count = gate_h * hidden;
  size_t b_count = gate_h;
  size_t h_bytes = (size_t)hidden * sizeof(TYPE);
  size_t x_bytes = (size_t)in_features * sizeof(TYPE);

  printf("Allocate device buffers\n");
  x_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, x_bytes, NULL, &_err));
  h_memobj[0] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, h_bytes, NULL, &_err));
  h_memobj[1] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, h_bytes, NULL, &_err));
  c_memobj[0] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, h_bytes, NULL, &_err));
  c_memobj[1] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, h_bytes, NULL, &_err));
  w_ih_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, w_ih_count * sizeof(TYPE), NULL, &_err));
  w_hh_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, w_hh_count * sizeof(TYPE), NULL, &_err));
  b_ih_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, b_count * sizeof(TYPE), NULL, &_err));
  b_hh_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, b_count * sizeof(TYPE), NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  cl_int build_err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
  if (build_err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 0) {
      std::vector<char> log(log_size + 1);
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
      log[log_size] = 0;
      printf("--- build log ---\n%s\n--- end ---\n", log.data());
    } else {
      printf("(empty build log)\n");
    }
    printf("clBuildProgram failed: %d\n", (int)build_err);
    cleanup();
    exit(-1);
  }

  kernel = CL_CHECK2(clCreateKernel(program, "lstm_cell", &_err));

  std::vector<TYPE> h_w_ih(w_ih_count), h_w_hh(w_hh_count), h_b_ih(b_count), h_b_hh(b_count);
  for (size_t i = 0; i < w_ih_count; ++i)
    h_w_ih[i] = (static_cast<float>(lcg_rand()) / RAND_MAX - 0.5f) * 0.5f;
  for (size_t i = 0; i < w_hh_count; ++i)
    h_w_hh[i] = (static_cast<float>(lcg_rand()) / RAND_MAX - 0.5f) * 0.5f;
  for (size_t i = 0; i < b_count; ++i) {
    h_b_ih[i] = (static_cast<float>(lcg_rand()) / RAND_MAX - 0.5f) * 0.2f;
    h_b_hh[i] = (static_cast<float>(lcg_rand()) / RAND_MAX - 0.5f) * 0.2f;
  }

  std::vector<TYPE> h_x((size_t)seq * in_features);
  for (size_t i = 0; i < h_x.size(); ++i)
    h_x[i] = (static_cast<float>(lcg_rand()) / RAND_MAX) * 2.0f - 1.0f;

  std::vector<TYPE> h_zero(hidden, 0);

  int a_in = in_features, a_hidden = hidden;
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x_memobj));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&w_ih_memobj));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&w_hh_memobj));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&b_ih_memobj));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&b_hh_memobj));
  CL_CHECK(clSetKernelArg(kernel, 9, sizeof(int), (void*)&a_in));
  CL_CHECK(clSetKernelArg(kernel, 10, sizeof(int), (void*)&a_hidden));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload weights and initial state\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, w_ih_memobj, CL_TRUE, 0,
                                w_ih_count * sizeof(TYPE), h_w_ih.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, w_hh_memobj, CL_TRUE, 0,
                                w_hh_count * sizeof(TYPE), h_w_hh.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_ih_memobj, CL_TRUE, 0,
                                b_count * sizeof(TYPE), h_b_ih.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_hh_memobj, CL_TRUE, 0,
                                b_count * sizeof(TYPE), h_b_hh.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, h_memobj[0], CL_TRUE, 0,
                                h_bytes, h_zero.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, c_memobj[0], CL_TRUE, 0,
                                h_bytes, h_zero.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t global_work_size[1] = {(size_t)((hidden + local_size - 1) / local_size * local_size)};
  size_t local_work_size[1] = {(size_t)local_size};

  auto time_start = std::chrono::high_resolution_clock::now();
  for (int step = 0; step < seq; ++step) {
    int cur = step & 1;
    int nxt = 1 - cur;

    CL_CHECK(clEnqueueWriteBuffer(commandQueue, x_memobj, CL_TRUE, 0, x_bytes,
                                  &h_x[(size_t)step * in_features], 0, NULL, NULL));
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&h_memobj[cur]));
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&c_memobj[cur]));
    CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&h_memobj[nxt]));
    CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&c_memobj[nxt]));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                    global_work_size, local_work_size, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  int final_slot = 1 - ((seq - 1) & 1);
  std::vector<TYPE> h_out(hidden, 0);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, h_memobj[final_slot], CL_TRUE, 0,
                               h_bytes, h_out.data(), 0, NULL, NULL));

  printf("Verify result\n");
  std::vector<TYPE> ref_h(hidden, 0), ref_c(hidden, 0);
  std::vector<TYPE> ref_h_next(hidden, 0), ref_c_next(hidden, 0);
  for (int step = 0; step < seq; ++step) {
    lstm_step_cpu(ref_h_next.data(), ref_c_next.data(), &h_x[(size_t)step * in_features],
                  ref_h.data(), ref_c.data(),
                  h_w_ih.data(), h_w_hh.data(), h_b_ih.data(), h_b_hh.data(),
                  in_features, hidden);
    ref_h = ref_h_next;
    ref_c = ref_c_next;
  }

  int errors = 0;
  for (int i = 0; i < hidden; ++i) {
    float diff = fabsf(h_out[i] - ref_h[i]);
    if (diff > 1e-5f + 1e-3f * fabsf(ref_h[i])) {
      if (errors < 100)
        printf("*** error: [%d] expected=%f, actual=%f\n", i, ref_h[i], h_out[i]);
      ++errors;
    }
  }
  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors != 0 ? 1 : 0;
}
