// MNIST_conv_tiny (Ventus-OpenCL-Testcase, _get_case/MNIST_conv_tiny) —
// standalone self-checking OpenCL port for Vortex.
//
// Three chained `conv` launches, the upstream network shape:
//   conv1: 1 -> 1,  5x5, stride 5    (n x n -> (n-5)/5+1 square), ReLU
//   conv2: 1 -> 1,  2x2, stride 1    (-> ...),                    ReLU
//   conv3: 1 -> 10, 4x4, stride 1    (-> 1x1),                    no ReLU
//
// Upstream: third_benchmark/Ventus-OpenCL-Testcase/_get_case/MNIST_conv_tiny/
// (kernel.cl is the upstream conv.cl). The upstream host loads PyTorch-produced
// weights and per-layer outputs from `data_gen/*.txt`; this port generates its
// own data with a deterministic LCG and computes the reference on the CPU, so
// no 65 MB of upstream data is carried. Every layer is compared, like upstream.
//
// Differences from the upstream host:
//   * weights/biases/input come from a private deterministic LCG in [-1,1)
//     (porting guide §13.2). Signed values make the ReLU branches fire.
//   * `-n` sets the input side (default 28, the upstream size); all other
//     shapes are derived with the upstream `(IH - K)/S + 1` rule.
//   * the kernel gained an `out_y/out_x` guard (see kernel.cl): the global
//     size is rounded up to the 4x4 work-group, and without the guard the extra
//     work-items would write past the end of the output buffer.
//   * comparison is |actual - expected| <= REL_TOL*|expected| + ABS_TOL.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#define KERNEL_NAME "conv"

#define REL_TOL 1e-3f
#define ABS_TOL 1e-3f

#define NUM_LAYERS 3

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

struct Layer {
  int ic, ih, iw;      // input channels / height / width
  int oc, kh, kw;      // output channels / kernel height / width
  int oh, ow;          // output height / width
  int relu, sh, sw;    // ReLU flag / strides
};

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

static int conv_out(int in, int k, int s) {
  return (in - k) / s + 1;
}

// Upstream network shape (see the MNIST_conv_tiny macros in main.cc upstream).
static std::vector<Layer> build_layers(int n) {
  std::vector<Layer> L;
  Layer l1;
  l1.ic = 1;  l1.ih = n;  l1.iw = n;
  l1.oc = 1;  l1.kh = 5;  l1.kw = 5;
  l1.sh = 5;  l1.sw = 5;  l1.relu = 1;
  l1.oh = conv_out(l1.ih, l1.kh, l1.sh);
  l1.ow = conv_out(l1.iw, l1.kw, l1.sw);
  L.push_back(l1);

  Layer l2;
  l2.ic = l1.oc;  l2.ih = l1.oh;  l2.iw = l1.ow;
  l2.oc = 1;      l2.kh = 2;      l2.kw = 2;
  l2.sh = 1;      l2.sw = 1;      l2.relu = 1;
  l2.oh = conv_out(l2.ih, l2.kh, l2.sh);
  l2.ow = conv_out(l2.iw, l2.kw, l2.sw);
  L.push_back(l2);

  Layer l3;
  l3.ic = l2.oc;  l3.ih = l2.oh;  l3.iw = l2.ow;
  l3.oc = 10;     l3.kh = 4;      l3.kw = 4;
  l3.sh = 1;      l3.sw = 1;      l3.relu = 0;
  l3.oh = conv_out(l3.ih, l3.kh, l3.sh);
  l3.ow = conv_out(l3.iw, l3.kw, l3.sw);
  L.push_back(l3);

  return L;
}

// Same loop nesting, accumulation order, boundary check and ReLU as kernel.cl.
static void conv_cpu(float* out, const float* in, const float* w, const float* b,
                     const Layer& L) {
  for (int oc = 0; oc < L.oc; ++oc) {
    for (int oy = 0; oy < L.oh; ++oy) {
      for (int ox = 0; ox < L.ow; ++ox) {
        float sum = b[oc];
        for (int ic = 0; ic < L.ic; ++ic) {
          for (int ky = 0; ky < L.kh; ++ky) {
            const int iy = oy * L.sh + ky;
            for (int kx = 0; kx < L.kw; ++kx) {
              const int ix = ox * L.sw + kx;
              if (iy >= L.ih || ix >= L.iw) continue;
              const int in_idx = ic * (L.ih * L.iw) + iy * L.iw + ix;
              const int w_idx = oc * (L.ic * L.kh * L.kw)
                              + ic * (L.kh * L.kw)
                              + ky * L.kw + kx;
              sum += in[in_idx] * w[w_idx];
            }
          }
        }
        if (L.relu && sum < 0) sum = 0;
        out[oc * (L.oh * L.ow) + oy * L.ow + ox] = sum;
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
cl_mem buf[NUM_LAYERS + 1];        // buf[0] = input, buf[i+1] = layer i output
cl_mem w_mem[NUM_LAYERS];
cl_mem b_mem[NUM_LAYERS];
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  for (int i = 0; i < NUM_LAYERS + 1; ++i)
    if (buf[i]) clReleaseMemObject(buf[i]);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    if (w_mem[i]) clReleaseMemObject(w_mem[i]);
    if (b_mem[i]) clReleaseMemObject(b_mem[i]);
  }
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int in_size = 28;

static void show_usage() {
  printf("Usage: [-n input_size] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      in_size = atoi(optarg);
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

  printf("float workload input=%dx%d\n", in_size, in_size);
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (in_size < 1) {
    printf("Error: input size must be at least 1\n");
    return -1;
  }

  std::vector<Layer> layers = build_layers(in_size);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    const Layer& L = layers[i];
    if (L.oh < 1 || L.ow < 1) {
      printf("Error: layer %d has an empty output (%dx%d); input size %d is too small\n",
             i + 1, L.oh, L.ow, in_size);
      return -1;
    }
  }

  const size_t in_n = (size_t)layers[0].ic * layers[0].ih * layers[0].iw;

  cl_platform_id platform_id;
  size_t kernel_size;

  // Getting platform and device information
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL,  &_err));

  printf("Allocate device buffers\n");
  buf[0] = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, in_n * sizeof(float), NULL, &_err));
  for (int i = 0; i < NUM_LAYERS; ++i) {
    const Layer& L = layers[i];
    const size_t out_n = (size_t)L.oc * L.oh * L.ow;
    const size_t w_n   = (size_t)L.oc * L.ic * L.kh * L.kw;
    buf[i + 1] = CL_CHECK2(clCreateBuffer(context, i + 1 < NUM_LAYERS ? CL_MEM_READ_WRITE : CL_MEM_WRITE_ONLY,
                                           out_n * sizeof(float), NULL, &_err));
    w_mem[i]   = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, w_n * sizeof(float), NULL, &_err));
    b_mem[i]   = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, L.oc * sizeof(float), NULL, &_err));
  }

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  // Host-side input and per-layer weights/biases.
  std::vector<float> h_in(in_n);
  for (size_t i = 0; i < in_n; ++i)
    h_in[i] = generate_pm1();

  std::vector<std::vector<float> > h_w(NUM_LAYERS), h_b(NUM_LAYERS);
  for (int i = 0; i < NUM_LAYERS; ++i) {
    const Layer& L = layers[i];
    h_w[i].resize((size_t)L.oc * L.ic * L.kh * L.kw);
    h_b[i].resize(L.oc);
    for (size_t j = 0; j < h_w[i].size(); ++j)
      h_w[i][j] = generate_pm1();
    for (int j = 0; j < L.oc; ++j)
      h_b[i][j] = generate_pm1();
  }

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, buf[0], CL_TRUE, 0, in_n * sizeof(float), h_in.data(), 0, NULL, NULL));
  for (int i = 0; i < NUM_LAYERS; ++i) {
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, w_mem[i], CL_TRUE, 0,
                                  h_w[i].size() * sizeof(float), h_w[i].data(), 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(commandQueue, b_mem[i], CL_TRUE, 0,
                                  h_b[i].size() * sizeof(float), h_b[i].data(), 0, NULL, NULL));
  }

  size_t local_work_size[3] = {1, 4, 4};

  // Run the layers in order; each one's CPU reference consumes the previous
  // layer's CPU reference output, and every layer is compared against device.
  std::vector<float> prev_ref = h_in;
  int errors = 0;

  for (int i = 0; i < NUM_LAYERS; ++i) {
    const Layer& L = layers[i];
    const size_t out_n = (size_t)L.oc * L.oh * L.ow;

    printf("Execute the kernel (layer %d: %d->%d, %dx%d, k%d s%d, relu=%d)\n",
           i + 1, L.ic, L.oc, L.ih, L.iw, L.kh, L.sh, L.relu);

    int arg = 0;
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&buf[i]));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&w_mem[i]));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&b_mem[i]));
    CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(cl_mem), (void *)&buf[i + 1]));
    int v;
    v = L.ic; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));
    v = L.ih; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));
    v = L.iw; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));
    v = L.kh; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));
    v = L.kw; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));
    v = L.oh; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));
    v = L.ow; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));
    v = L.relu; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));
    v = L.sh; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));
    v = L.sw; CL_CHECK(clSetKernelArg(kernel, arg++, sizeof(int), (void *)&v));

    size_t global_work_size[3];
    global_work_size[0] = round_up((size_t)L.oc, local_work_size[0]);
    global_work_size[1] = round_up((size_t)L.oh, local_work_size[1]);
    global_work_size[2] = round_up((size_t)L.ow, local_work_size[2]);

    auto time_start = std::chrono::high_resolution_clock::now();
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, global_work_size, local_work_size, 0, NULL, NULL));
    CL_CHECK(clFinish(commandQueue));
    auto time_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
    printf("Elapsed time: %lg ms\n", elapsed);

    printf("Download destination buffer\n");
    std::vector<float> h_out(out_n, 0.0f);
    CL_CHECK(clEnqueueReadBuffer(commandQueue, buf[i + 1], CL_TRUE, 0, out_n * sizeof(float), h_out.data(), 0, NULL, NULL));

    printf("Verify result\n");
    std::vector<float> h_ref(out_n);
    conv_cpu(h_ref.data(), prev_ref.data(), h_w[i].data(), h_b[i].data(), L);

    for (size_t j = 0; j < out_n; ++j) {
      float a = h_out[j];
      float b = h_ref[j];
      float tol = REL_TOL * fabsf(b) + ABS_TOL;
      if (fabsf(a - b) > tol) {
        if (errors < 100) {
          printf("*** error: layer%d[%d] expected=%f, actual=%f (diff=%g)\n",
                 i + 1, (int)j, b, a, fabsf(a - b));
        }
        ++errors;
      }
    }

    prev_ref.swap(h_ref);
  }

  if (0 == errors) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }

  cleanup();

  return errors;
}