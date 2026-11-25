// Blue Porcelain basic/NQUEEN — standalone self-checking OpenCL port for Vortex.
//
// Upstream: third_benchmark/blue-porcelain/benchmarks/basic/NQUEEN/nqueen.cu
//
// The host keeps upstream's structure: enumerate the start conditions up to
// `host_mark` rows, hand them to the device, and sum the completions. Upstream's
// serial `solve_nqueen()` is the reference. Its fixed 96-thread "steps" batching
// is replaced by launching exactly ceil(conditions/16) work-groups, which is the
// same computation without the idle groups.
//
// Default n is 8 (upstream default), which has 92 solutions.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <string.h>
#include <unistd.h>
#include <chrono>
#include <vector>
#include <stdint.h>

#define GROUP_SIZE 16

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

  fseek(fp , 0 , SEEK_END);
  long fsize = ftell(fp);
  rewind(fp);

  *data = (uint8_t*)malloc(fsize);
  *size = fread(*data, 1, fsize, fp);

  fclose(fp);

  return 0;
}

// Upstream's serial non-recursive solver, used as the reference.
static long long solve_nqueen_cpu(int n) {
  unsigned int mask[32];
  unsigned int l_mask[32];
  unsigned int r_mask[32];
  unsigned int m[32];

  if (n <= 0 || n > 31)
    return 0;

  const unsigned int t_mask = (1u << n) - 1;
  long long total = 0;
  long long upper_total = 0;
  int i = 0, j;
  unsigned int index;

  mask[0] = l_mask[0] = r_mask[0] = m[0] = 0;

  for (j = 0; j < (n + 1) / 2; j++) {
    index = 1u << j;
    m[0] |= index;

    mask[1] = index;
    l_mask[1] = index << 1;
    r_mask[1] = index >> 1;
    m[1] = mask[1] | l_mask[1] | r_mask[1];
    i = 1;

    if (n % 2 == 1 && j == (n + 1) / 2 - 1) {
      upper_total = total;
      total = 0;
    }

    while (i > 0) {
      if ((m[i] & t_mask) == t_mask) {
        i--;
      } else {
        index = (m[i] + 1) & ~m[i];
        m[i] |= index;
        if ((index & t_mask) != 0) {
          if (i + 1 == n) {
            total++;
            i--;
          } else {
            mask[i + 1] = mask[i] | index;
            l_mask[i + 1] = (l_mask[i] | index) << 1;
            r_mask[i + 1] = (r_mask[i] | index) >> 1;
            m[i + 1] = mask[i + 1] | l_mask[i + 1] | r_mask[i + 1];
            i++;
          }
        } else {
          i--;
        }
      }
    }
  }

  if (n % 2 == 0)
    return total * 2;
  return upper_total * 2 + total;
}

struct Condition {
  uint32_t mask;
  uint32_t l_mask;
  uint32_t r_mask;
};

// Enumerate the states reached after placing a queen in `first_col` and
// recursing to depth `mark` (upstream's host loop, one first-column at a time).
static void collect_conditions(int n, int first_col, int mark,
                               std::vector<Condition>& out) {
  unsigned int mask[32];
  unsigned int l_mask[32];
  unsigned int r_mask[32];
  unsigned int m[32];

  const unsigned int t_mask = (1u << n) - 1;

  mask[0] = l_mask[0] = r_mask[0] = m[0] = 0;

  unsigned int index = 1u << first_col;
  m[0] |= index;

  mask[1] = index;
  l_mask[1] = index << 1;
  r_mask[1] = index >> 1;
  m[1] = mask[1] | l_mask[1] | r_mask[1];
  int i = 1;

  while (i > 0) {
    if ((m[i] & t_mask) == t_mask) {
      i--;
    } else {
      index = (m[i] + 1) & ~m[i];
      m[i] |= index;
      if ((index & t_mask) != 0) {
        mask[i + 1] = mask[i] | index;
        l_mask[i + 1] = (l_mask[i] | index) << 1;
        r_mask[i + 1] = (r_mask[i] | index) >> 1;
        m[i + 1] = mask[i + 1] | l_mask[i + 1] | r_mask[i + 1];
        i++;
        if (i == mark) {
          out.push_back({mask[i], l_mask[i], r_mask[i]});
          i--;
        }
      } else {
        i--;
      }
    }
  }
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);
  if (kernel_bin) free(kernel_bin);
}

static long long run_on_device(const std::vector<Condition>& conds,
                               int n, int kernel_mark) {
  if (conds.empty())
    return 0;

  uint32_t count = (uint32_t)conds.size();
  uint32_t grid = (count + GROUP_SIZE - 1) / GROUP_SIZE;
  size_t mask_nbytes = (size_t)count * sizeof(uint32_t);
  size_t res_nbytes = (size_t)grid * sizeof(uint32_t);

  std::vector<uint32_t> h_mask(count), h_l(count), h_r(count);
  for (uint32_t i = 0; i < count; ++i) {
    h_mask[i] = conds[i].mask;
    h_l[i] = conds[i].l_mask;
    h_r[i] = conds[i].r_mask;
  }

  cl_mem mask_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, mask_nbytes, NULL, &_err));
  cl_mem l_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, mask_nbytes, NULL, &_err));
  cl_mem r_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, mask_nbytes, NULL, &_err));
  cl_mem res_mem = CL_CHECK2(clCreateBuffer(context, CL_MEM_WRITE_ONLY, res_nbytes, NULL, &_err));

  CL_CHECK(clEnqueueWriteBuffer(commandQueue, mask_mem, CL_TRUE, 0, mask_nbytes, h_mask.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, l_mem, CL_TRUE, 0, mask_nbytes, h_l.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, r_mem, CL_TRUE, 0, mask_nbytes, h_r.data(), 0, NULL, NULL));

  int total_conditions = (int)count;
  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mask_mem));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&l_mem));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&r_mem));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&res_mem));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), &n));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), &kernel_mark));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), &total_conditions));

  size_t local_size = GROUP_SIZE;
  size_t global_size = (size_t)grid * GROUP_SIZE;
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));

  std::vector<uint32_t> h_res(grid, 0);
  CL_CHECK(clEnqueueReadBuffer(commandQueue, res_mem, CL_TRUE, 0, res_nbytes, h_res.data(), 0, NULL, NULL));

  long long total = 0;
  for (uint32_t i = 0; i < grid; ++i)
    total += h_res[i];

  clReleaseMemObject(mask_mem);
  clReleaseMemObject(l_mem);
  clReleaseMemObject(r_mem);
  clReleaseMemObject(res_mem);

  return total;
}

int size = 8;

static void show_usage() {
  printf("Usage: [-n queens] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
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
}

int main (int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 2 || size > 31) {
    printf("Invalid n=%d: must be between 2 and 31\n", size);
    return 1;
  }

  printf("N-queen n=%d\n", size);

  int n = size;
  int host_mark = (n > 11) ? (n - 10) : 2;
  int kernel_mark = n - host_mark;

  cl_platform_id platform_id;
  size_t kernel_size;

  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Create program from kernel source\n");
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  if (program == NULL) {
    cleanup();
    return -1;
  }
  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, "solve_nqueen", &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Execute the kernel\n");
  auto time_start = std::chrono::high_resolution_clock::now();

  long long device_total = 0;
  std::vector<Condition> conds;

  // Even-symmetry half: first queen in columns [0, n/2).
  for (int j = 0; j < n / 2; ++j) {
    conds.clear();
    collect_conditions(n, j, host_mark, conds);
    device_total += run_on_device(conds, n, kernel_mark);
  }
  device_total *= 2;

  // Odd n: the middle column is its own mirror image, counted once.
  if (n % 2 == 1) {
    conds.clear();
    collect_conditions(n, (n - 1) / 2, host_mark, conds);
    device_total += run_on_device(conds, n, kernel_mark);
  }

  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Verify result\n");
  long long ref = solve_nqueen_cpu(n);

  int errors = 0;
  if (device_total != ref) {
    printf("*** error: expected=%lld, actual=%lld\n", ref, device_total);
    ++errors;
  }

  if (errors != 0)
    printf("FAILED! - %d errors\n", errors);
  else
    printf("PASSED!\n");

  cleanup();

  return errors;
}