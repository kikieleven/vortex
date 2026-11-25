// Particle filter (single-precision "optimized" variant) — standalone
// self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/gpu-rodinia/opencl/particlefilter (the
// particle_single variant + ex_particle_OCL_single_seq.cpp). The upstream
// double-precision variants cannot run here: the Vortex OpenCL device has no
// cl_khr_fp64 (CL_DEVICE_DOUBLE_FP_CONFIG == 0).
//
// Per frame the device runs the same four kernels as upstream:
//   likelihood_kernel       -> per-particle likelihood, weights, block sum
//   sum_kernel              -> sum the block sums
//   normalize_weights_kernel-> normalize weights, build CDF and u[]
//   find_index_kernel       -> systematic resampling into xj/yj
//
// Vortex mapping: ONE work-group of 16 work-items, Nparticles == 16. Upstream
// assumes a single block too (normalize_weights_kernel writes u[0] in
// work-item 0 and the others consume it; OpenCL barriers do not synchronize
// across work-groups). The input video is generated with the upstream
// videoSequence() and a private LCG (never rand(), which PoCL also consumes).
// The final particle state is checked against a serial CPU reference running
// the identical single-precision arithmetic and seed sequence.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <limits.h>
#include <CL/opencl.h>

#define CL_CHECK(_expr)                                                \
  do {                                                                 \
    cl_int _err = _expr;                                               \
    if (_err == CL_SUCCESS)                                            \
      break;                                                           \
    printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);    \
    cleanup();                                                         \
    exit(-1);                                                          \
  } while (0)

#define CL_CHECK2(_expr)                                               \
  ({                                                                   \
    cl_int _err = CL_INVALID_VALUE;                                    \
    decltype(_expr) _ret = _expr;                                      \
    if (_err != CL_SUCCESS) {                                          \
      printf("OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);  \
      cleanup();                                                       \
      exit(-1);                                                        \
    }                                                                  \
    _ret;                                                              \
  })

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

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel likelihood_kernel = NULL;
cl_kernel sum_kernel = NULL;
cl_kernel normalize_weights_kernel = NULL;
cl_kernel find_index_kernel = NULL;
cl_mem d_arrayX = NULL, d_arrayY = NULL, d_xj = NULL, d_yj = NULL;
cl_mem d_CDF = NULL, d_u = NULL, d_likelihood = NULL, d_weights = NULL;
cl_mem d_ind = NULL, d_objxy = NULL, d_seed = NULL, d_partial_sums = NULL;
cl_mem d_I = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (likelihood_kernel) clReleaseKernel(likelihood_kernel);
  if (sum_kernel) clReleaseKernel(sum_kernel);
  if (normalize_weights_kernel) clReleaseKernel(normalize_weights_kernel);
  if (find_index_kernel) clReleaseKernel(find_index_kernel);
  if (program) clReleaseProgram(program);
  if (d_arrayX) clReleaseMemObject(d_arrayX);
  if (d_arrayY) clReleaseMemObject(d_arrayY);
  if (d_xj) clReleaseMemObject(d_xj);
  if (d_yj) clReleaseMemObject(d_yj);
  if (d_CDF) clReleaseMemObject(d_CDF);
  if (d_u) clReleaseMemObject(d_u);
  if (d_likelihood) clReleaseMemObject(d_likelihood);
  if (d_weights) clReleaseMemObject(d_weights);
  if (d_ind) clReleaseMemObject(d_ind);
  if (d_objxy) clReleaseMemObject(d_objxy);
  if (d_seed) clReleaseMemObject(d_seed);
  if (d_partial_sums) clReleaseMemObject(d_partial_sums);
  if (d_I) clReleaseMemObject(d_I);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

// ---------------------------------------------------------------------------
// Upstream constants
// ---------------------------------------------------------------------------
#define PF_SCALE_FACTOR 300
#define PF_LOCAL_SIZE 16
#define LCG_A 1103515245u
#define LCG_C 12345u
#define LCG_M 2147483647
static const double PI_D = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Upstream host helpers (video generation). These only produce the input, so
// they reproduce the upstream double-trig host variant; the device and the CPU
// reference both start from the resulting bytes.
// ---------------------------------------------------------------------------
static int lcg_next(int s) {
  return (int)(LCG_A * (uint32_t)s + LCG_C);
}

static float host_randu(int* seed, int index) {
  int num = lcg_next(seed[index]);
  seed[index] = num % LCG_M;
  return fabsf((float)seed[index] / (float)LCG_M);
}

static float host_randn(int* seed, int index) {
  float u = host_randu(seed, index);
  float v = host_randu(seed, index);
  float cosine = (float)cos(2.0 * PI_D * (double)v);
  float rt = (float)(-2.0 * log((double)u));
  return (float)sqrt((double)rt) * cosine;
}

static float round_float(float value) {
  int newValue = (int)(value);
  if (value - newValue < .5f)
    return (float)newValue;
  else
    return (float)(newValue++);
}

static void set_if(int testValue, int newValue, unsigned char* array3D,
                   int dimX, int dimY, int dimZ) {
  for (int x = 0; x < dimX; ++x)
    for (int y = 0; y < dimY; ++y)
      for (int z = 0; z < dimZ; ++z)
        if (array3D[x * dimY * dimZ + y * dimZ + z] == testValue)
          array3D[x * dimY * dimZ + y * dimZ + z] = (unsigned char)newValue;
}

static void add_noise(unsigned char* array3D, int dimX, int dimY, int dimZ,
                      int* seed) {
  for (int x = 0; x < dimX; ++x)
    for (int y = 0; y < dimY; ++y)
      for (int z = 0; z < dimZ; ++z) {
        int idx = x * dimY * dimZ + y * dimZ + z;
        unsigned char noise =
            (unsigned char)((int)(5 * host_randn(seed, 0)));
        array3D[idx] = (unsigned char)((int)array3D[idx] + (int)noise);
      }
}

static void strel_disk(int* disk, int radius) {
  int diameter = radius * 2 - 1;
  for (int x = 0; x < diameter; ++x)
    for (int y = 0; y < diameter; ++y) {
      float distance = sqrtf((float)((x - radius + 1) * (x - radius + 1)) +
                             (float)((y - radius + 1) * (y - radius + 1)));
      disk[x * diameter + y] = (distance < radius) ? 1 : 0;
    }
}

static void dilate_matrix(unsigned char* matrix, int posX, int posY, int posZ,
                          int dimX, int dimY, int dimZ, int error) {
  int startX = posX - error;
  while (startX < 0) startX++;
  int startY = posY - error;
  while (startY < 0) startY++;
  int endX = posX + error;
  while (endX > dimX) endX--;
  int endY = posY + error;
  while (endY > dimY) endY--;
  for (int x = startX; x < endX; ++x)
    for (int y = startY; y < endY; ++y) {
      float distance = sqrtf((float)((x - posX) * (x - posX)) +
                             (float)((y - posY) * (y - posY)));
      if (distance < error)
        matrix[x * dimY * dimZ + y * dimZ + posZ] = 1;
    }
}

static void imdilate_disk(unsigned char* matrix, int dimX, int dimY, int dimZ,
                          int error, unsigned char* newMatrix) {
  for (int z = 0; z < dimZ; ++z)
    for (int x = 0; x < dimX; ++x)
      for (int y = 0; y < dimY; ++y)
        if (matrix[x * dimY * dimZ + y * dimZ + z] == 1)
          dilate_matrix(newMatrix, x, y, z, dimX, dimY, dimZ, error);
}

static void getneighbors(const int* se, int numOnes, int* neighbors, int radius) {
  int neighY = 0;
  int center = radius - 1;
  int diameter = radius * 2 - 1;
  for (int x = 0; x < diameter; ++x)
    for (int y = 0; y < diameter; ++y)
      if (se[x * diameter + y]) {
        neighbors[neighY * 2] = (int)(y - center);
        neighbors[neighY * 2 + 1] = (int)(x - center);
        neighY++;
      }
}

static void video_sequence(unsigned char* I, int IszX, int IszY, int Nfr,
                           int* seed) {
  int max_size = IszX * IszY * Nfr;
  int x0 = (int)round_float((float)(IszY / 2.0));
  int y0 = (int)round_float((float)(IszX / 2.0));
  I[x0 * IszY * Nfr + y0 * Nfr + 0] = 1;

  for (int k = 1; k < Nfr; ++k) {
    int xk = abs(x0 + (k - 1));
    int yk = abs(y0 - 2 * (k - 1));
    int pos = yk * IszY * Nfr + xk * Nfr + k;
    if (pos >= max_size)
      pos = 0;
    I[pos] = 1;
  }

  unsigned char* newMatrix =
      (unsigned char*)calloc(IszX * IszY * Nfr, sizeof(unsigned char));
  imdilate_disk(I, IszX, IszY, Nfr, 5, newMatrix);
  for (int x = 0; x < IszX; ++x)
    for (int y = 0; y < IszY; ++y)
      for (int k = 0; k < Nfr; ++k)
        I[x * IszY * Nfr + y * Nfr + k] = newMatrix[x * IszY * Nfr + y * Nfr + k];
  free(newMatrix);

  set_if(0, 100, I, IszX, IszY, Nfr);
  set_if(1, 228, I, IszX, IszY, Nfr);
  add_noise(I, IszX, IszY, Nfr, seed);
}

// ---------------------------------------------------------------------------
// CPU reference: the device kernels replayed serially with the same float ops
// and the same seed consumption order.
// ---------------------------------------------------------------------------
// Device-matching RNG (float trig, as in particle_single.cl).
static float ref_randu(int* seed, int index) {
  int num = lcg_next(seed[index]);
  seed[index] = num % LCG_M;
  return fabsf((float)seed[index] / (float)LCG_M);
}

static float ref_randn(int* seed, int index) {
  float pi = 3.14159265358979323846f;
  float u = ref_randu(seed, index);
  float v = ref_randu(seed, index);
  float cosine = cosf(2.0f * pi * v);
  float rt = -2.0f * logf(u);
  return sqrtf(rt) * cosine;
}

// Upstream dev_round_float truncates (the post-increment returns the old value).
static float dev_round_float(float value) {
  return (float)((int)value);
}

struct PfState {
  std::vector<float> arrayX, arrayY, xj, yj, CDF, u, likelihood, weights;
  std::vector<int> ind, seed;
  std::vector<float> partial_sums;
};

static void ref_likelihood(PfState& s, const std::vector<unsigned char>& I,
                           const std::vector<int>& objxy, int Nparticles,
                           int countOnes, int max_size, int k, int IszY,
                           int Nfr) {
  for (int i = 0; i < Nparticles; ++i) {
    s.arrayX[i] = s.xj[i];
    s.arrayY[i] = s.yj[i];
    s.weights[i] = 1.0f / (float)Nparticles;

    s.arrayX[i] = s.arrayX[i] + 1.0f + 5.0f * ref_randn(s.seed.data(), i);
    s.arrayY[i] = s.arrayY[i] - 2.0f + 2.0f * ref_randn(s.seed.data(), i);
  }

  for (int i = 0; i < Nparticles; ++i) {
    for (int y = 0; y < countOnes; ++y) {
      int indX = (int)dev_round_float(s.arrayX[i]) + objxy[y * 2 + 1];
      int indY = (int)dev_round_float(s.arrayY[i]) + objxy[y * 2];
      int v = abs(indX * IszY * Nfr + indY * Nfr + k);
      if (v >= max_size) v = 0;
      s.ind[i * countOnes + y] = v;
    }
    float sum = 0.0f;
    for (int x = 0; x < countOnes; ++x) {
      float q = (float)((int)I[s.ind[i * countOnes + x]] - 100);
      float r = (float)((int)I[s.ind[i * countOnes + x]] - 228);
      sum += (q * q - r * r) / 50.0f;
    }
    s.likelihood[i] = sum;
    s.likelihood[i] = s.likelihood[i] / (float)countOnes - (float)PF_SCALE_FACTOR;
    s.weights[i] = s.weights[i] * expf(s.likelihood[i]);
  }

  float buffer[PF_LOCAL_SIZE];
  for (int t = 0; t < PF_LOCAL_SIZE; ++t) buffer[t] = 0.0f;
  for (int i = 0; i < Nparticles; ++i) buffer[i] = s.weights[i];
  for (int step = PF_LOCAL_SIZE / 2; step > 0; step >>= 1)
    for (int t = 0; t < step; ++t) buffer[t] += buffer[t + step];
  s.partial_sums[0] = buffer[0];
}

static void ref_sum(PfState& s, int Nparticles) {
  const int num_blocks = (Nparticles + PF_LOCAL_SIZE - 1) / PF_LOCAL_SIZE;
  float sum = 0.0f;
  for (int x = 0; x < num_blocks; ++x) sum += s.partial_sums[x];
  s.partial_sums[0] = sum;
}

static void ref_normalize(PfState& s, int Nparticles) {
  float sumWeights = s.partial_sums[0];
  for (int i = 0; i < Nparticles; ++i)
    s.weights[i] = s.weights[i] / sumWeights;

  s.CDF[0] = s.weights[0];
  for (int x = 1; x < Nparticles; ++x)
    s.CDF[x] = s.weights[x] + s.CDF[x - 1];
  s.u[0] = (1.0f / (float)Nparticles) * ref_randu(s.seed.data(), 0);

  float u1 = s.u[0];
  for (int i = 0; i < Nparticles; ++i)
    s.u[i] = u1 + (float)i / (float)Nparticles;
}

static void ref_find_index(PfState& s, int Nparticles) {
  for (int i = 0; i < Nparticles; ++i) {
    int index = -1;
    for (int x = 0; x < Nparticles; ++x)
      if (s.CDF[x] >= s.u[i]) { index = x; break; }
    if (index == -1) index = Nparticles - 1;
    s.xj[i] = s.arrayX[index];
    s.yj[i] = s.arrayY[index];
  }
}

// ---------------------------------------------------------------------------
// Workload parameters
// ---------------------------------------------------------------------------
static int dimX = 32;
static int dimY = 32;
static int Nfr = 4;
static int Nparticles = 16;

static void show_usage() {
  printf("Usage: [-x dimX] [-y dimY] [-f frames] [-p particles] [-h]\n"
         "  particles must be <= %d and equal to the work-group size.\n",
         PF_LOCAL_SIZE);
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "x:y:f:p:h")) != -1) {
    switch (c) {
    case 'x': dimX = atoi(optarg); break;
    case 'y': dimY = atoi(optarg); break;
    case 'f': Nfr = atoi(optarg); break;
    case 'p': Nparticles = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default: show_usage(); exit(-1);
    }
  }
  if (dimX < 4 || dimY < 4 || Nfr < 2 || Nparticles < 2) {
    printf("Error: need dimX,dimY >= 4, frames >= 2, particles >= 2\n");
    exit(-1);
  }
  if (Nparticles != PF_LOCAL_SIZE) {
    printf("Error: particles (%d) must equal the work-group size (%d)\n",
           Nparticles, PF_LOCAL_SIZE);
    exit(-1);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);
  const int max_size = dimX * dimY * Nfr;
  printf("ParticleFilter: dimX=%d dimY=%d frames=%d particles=%d\n",
         dimX, dimY, Nfr, Nparticles);

  // Disk structuring element (upstream radius 5 -> 69 ones).
  const int radius = 5;
  const int diameter = radius * 2 - 1;
  std::vector<int> disk(diameter * diameter, 0);
  strel_disk(disk.data(), radius);
  int countOnes = 0;
  for (int i = 0; i < diameter * diameter; ++i)
    if (disk[i] == 1) countOnes++;
  std::vector<int> objxy(countOnes * 2, 0);
  getneighbors(disk.data(), countOnes, objxy.data(), radius);
  printf("Disk: countOnes=%d\n", countOnes);

  // Deterministic video, upstream seed scheme (seed[i] = i+1).
  std::vector<int> seed(Nparticles);
  for (int i = 0; i < Nparticles; ++i) seed[i] = i + 1;
  std::vector<unsigned char> video(max_size, 0);
  video_sequence(video.data(), dimX, dimY, Nfr, seed.data());

  // Initial particle state (upstream init).
  const float xe = round_float((float)(dimY / 2.0));
  const float ye = round_float((float)(dimX / 2.0));
  std::vector<float> init_xj(Nparticles, xe), init_yj(Nparticles, ye);
  std::vector<float> init_weights(Nparticles, 1.0f / (float)Nparticles);
  std::vector<int> init_seed(seed);  // after video generation, as upstream

  printf("Create context\n");
  cl_platform_id platform_id;
  size_t kernel_size;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  printf("Allocate device buffers\n");
  const size_t float_bytes = sizeof(float) * Nparticles;
  d_arrayX = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, float_bytes, NULL, &_err));
  d_arrayY = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, float_bytes, NULL, &_err));
  d_xj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, float_bytes, NULL, &_err));
  d_yj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, float_bytes, NULL, &_err));
  d_CDF = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, float_bytes, NULL, &_err));
  d_u = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, float_bytes, NULL, &_err));
  d_likelihood = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, float_bytes, NULL, &_err));
  d_weights = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, float_bytes, NULL, &_err));
  d_partial_sums = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, float_bytes, NULL, &_err));
  d_ind = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   sizeof(int) * Nparticles * countOnes, NULL, &_err));
  d_objxy = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     sizeof(int) * 2 * countOnes, NULL, &_err));
  d_seed = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(int) * Nparticles, NULL, &_err));
  d_I = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY,
                                 sizeof(unsigned char) * max_size, NULL, &_err));

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
      context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  char build_opts[64];
  snprintf(build_opts, sizeof(build_opts), "-DLOCAL_SIZE=%d", PF_LOCAL_SIZE);
  CL_CHECK(clBuildProgram(program, 1, &device_id, build_opts, NULL, NULL));

  likelihood_kernel = CL_CHECK2(clCreateKernel(program, "likelihood_kernel", &_err));
  sum_kernel = CL_CHECK2(clCreateKernel(program, "sum_kernel", &_err));
  normalize_weights_kernel =
      CL_CHECK2(clCreateKernel(program, "normalize_weights_kernel", &_err));
  find_index_kernel = CL_CHECK2(clCreateKernel(program, "find_index_kernel", &_err));
  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_I, CL_TRUE, 0,
                                sizeof(unsigned char) * max_size, video.data(),
                                0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_objxy, CL_TRUE, 0,
                                sizeof(int) * 2 * countOnes, objxy.data(),
                                0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_weights, CL_TRUE, 0, float_bytes,
                                init_weights.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_xj, CL_TRUE, 0, float_bytes,
                                init_xj.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_yj, CL_TRUE, 0, float_bytes,
                                init_yj.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, d_seed, CL_TRUE, 0,
                                sizeof(int) * Nparticles, init_seed.data(),
                                0, NULL, NULL));

  size_t local_work_size = PF_LOCAL_SIZE;
  size_t global_work_size = PF_LOCAL_SIZE;

  // Static likelihood arguments.
  CL_CHECK(clSetKernelArg(likelihood_kernel, 0, sizeof(cl_mem), &d_arrayX));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 1, sizeof(cl_mem), &d_arrayY));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 2, sizeof(cl_mem), &d_xj));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 3, sizeof(cl_mem), &d_yj));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 4, sizeof(cl_mem), &d_CDF));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 5, sizeof(cl_mem), &d_ind));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 6, sizeof(cl_mem), &d_objxy));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 7, sizeof(cl_mem), &d_likelihood));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 8, sizeof(cl_mem), &d_I));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 9, sizeof(cl_mem), &d_u));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 10, sizeof(cl_mem), &d_weights));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 11, sizeof(int), &Nparticles));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 12, sizeof(int), &countOnes));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 13, sizeof(int), &max_size));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 15, sizeof(int), &dimY));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 16, sizeof(int), &Nfr));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 17, sizeof(cl_mem), &d_seed));
  CL_CHECK(clSetKernelArg(likelihood_kernel, 18, sizeof(cl_mem), &d_partial_sums));

  CL_CHECK(clSetKernelArg(sum_kernel, 0, sizeof(cl_mem), &d_partial_sums));
  CL_CHECK(clSetKernelArg(sum_kernel, 1, sizeof(int), &Nparticles));

  CL_CHECK(clSetKernelArg(normalize_weights_kernel, 0, sizeof(cl_mem), &d_weights));
  CL_CHECK(clSetKernelArg(normalize_weights_kernel, 1, sizeof(int), &Nparticles));
  CL_CHECK(clSetKernelArg(normalize_weights_kernel, 2, sizeof(cl_mem), &d_partial_sums));
  CL_CHECK(clSetKernelArg(normalize_weights_kernel, 3, sizeof(cl_mem), &d_CDF));
  CL_CHECK(clSetKernelArg(normalize_weights_kernel, 4, sizeof(cl_mem), &d_u));
  CL_CHECK(clSetKernelArg(normalize_weights_kernel, 5, sizeof(cl_mem), &d_seed));

  CL_CHECK(clSetKernelArg(find_index_kernel, 0, sizeof(cl_mem), &d_arrayX));
  CL_CHECK(clSetKernelArg(find_index_kernel, 1, sizeof(cl_mem), &d_arrayY));
  CL_CHECK(clSetKernelArg(find_index_kernel, 2, sizeof(cl_mem), &d_CDF));
  CL_CHECK(clSetKernelArg(find_index_kernel, 3, sizeof(cl_mem), &d_u));
  CL_CHECK(clSetKernelArg(find_index_kernel, 4, sizeof(cl_mem), &d_xj));
  CL_CHECK(clSetKernelArg(find_index_kernel, 5, sizeof(cl_mem), &d_yj));
  CL_CHECK(clSetKernelArg(find_index_kernel, 6, sizeof(int), &Nparticles));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();
  for (int k = 1; k < Nfr; ++k) {
    CL_CHECK(clSetKernelArg(likelihood_kernel, 14, sizeof(int), &k));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, likelihood_kernel, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, sum_kernel, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, normalize_weights_kernel, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, find_index_kernel, 1, NULL,
                                    &global_work_size, &local_work_size, 0, NULL, NULL));
  }
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  std::vector<float> h_xj(Nparticles), h_yj(Nparticles), h_arrayX(Nparticles),
      h_arrayY(Nparticles), h_weights(Nparticles), h_CDF(Nparticles),
      h_u(Nparticles);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_xj, CL_TRUE, 0, float_bytes, h_xj.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_yj, CL_TRUE, 0, float_bytes, h_yj.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_arrayX, CL_TRUE, 0, float_bytes, h_arrayX.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_arrayY, CL_TRUE, 0, float_bytes, h_arrayY.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_weights, CL_TRUE, 0, float_bytes, h_weights.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_CDF, CL_TRUE, 0, float_bytes, h_CDF.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, d_u, CL_TRUE, 0, float_bytes, h_u.data(), 0, NULL, NULL));

  printf("Verify result\n");
  PfState ref;
  ref.arrayX.assign(Nparticles, 0.0f);
  ref.arrayY.assign(Nparticles, 0.0f);
  ref.xj = init_xj;
  ref.yj = init_yj;
  ref.CDF.assign(Nparticles, 0.0f);
  ref.u.assign(Nparticles, 0.0f);
  ref.likelihood.assign(Nparticles, 0.0f);
  ref.weights = init_weights;
  ref.ind.assign(Nparticles * countOnes, 0);
  ref.seed = init_seed;
  ref.partial_sums.assign(Nparticles, 0.0f);

  for (int k = 1; k < Nfr; ++k) {
    ref_likelihood(ref, video, objxy, Nparticles, countOnes, max_size, k,
                   dimY, Nfr);
    ref_sum(ref, Nparticles);
    ref_normalize(ref, Nparticles);
    ref_find_index(ref, Nparticles);
  }

  const float atol = 1e-3f;
  const float rtol = 1e-3f;
  int errors = 0;

  struct Check {
    const char* name;
    const std::vector<float>& ref;
    const std::vector<float>& got;
  } checks[] = {
      {"xj", ref.xj, h_xj},
      {"yj", ref.yj, h_yj},
      {"arrayX", ref.arrayX, h_arrayX},
      {"arrayY", ref.arrayY, h_arrayY},
      {"weights", ref.weights, h_weights},
      {"CDF", ref.CDF, h_CDF},
      {"u", ref.u, h_u},
  };

  for (const auto& ch : checks) {
    int bad = 0;
    for (int i = 0; i < Nparticles; ++i) {
      float a = ch.ref[i], b = ch.got[i];
      if (fabsf(a - b) > atol + rtol * fabsf(a)) {
        if (errors < 100)
          printf("*** error: %s[%d] expected=%f, actual=%f\n", ch.name, i, a, b);
        ++errors;
        ++bad;
      }
    }
    printf("  %-8s : %s\n", ch.name, bad == 0 ? "OK" : "MISMATCH");
  }

  cleanup();
  if (errors != 0) {
    printf("FAILED! - %d errors\n", errors);
    return errors;
  }
  printf("PASSED!\n");
  return 0;
}