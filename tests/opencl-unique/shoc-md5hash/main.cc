// SHOC MD5 hash (level1/md5hash) — standalone self-checking OpenCL port for
// Vortex.
//
// Source:
//   third_benchmark/shoc/src/opencl/level1/md5hash/md5.cl
//   third_benchmark/shoc/src/opencl/level1/md5hash/MD5Hash.cpp:188-350
//
// SHOC searches a keyspace for a key whose MD5 equals a caller-supplied digest
// (FindKeyWithDigest_Kernel). This port keeps that structure: the host picks
// index 0 of the keyspace, computes its digest with the same MD5 core, and the
// kernel must find that index/key/digest.
//
// Integer algorithm, so the comparison is exact. No precision switch exists in
// SHOC for this case (it is pure integer), so none is added here.
//
// The MD5 core is shared with the kernel through md5_core.h. Because a shared
// bug would cancel out in the GPU-vs-CPU comparison, main() also checks the
// core against the RFC 1321 test vector for "abc" before running the kernel.

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>

#include "md5_core.h"

#define KERNEL_NAME "FindKeyWithDigest_Kernel"

// Vortex's OpenCL device caps a work-group at 16 work-items.
#define LWS 16

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

// RFC 1321 MD5("abc"), as four little-endian 32-bit words.
static const unsigned int MD5_ABC[4] = {
  0x98500190u, 0xb04fd23cu, 0x7d3f96d6u, 0x727fe128u
};

static bool check_md5_core_against_abc() {
  union { unsigned int words[2]; unsigned char bytes[8]; } key;
  key.words[0] = 0;
  key.words[1] = 0;
  key.bytes[0] = 'a';
  key.bytes[1] = 'b';
  key.bytes[2] = 'c';

  unsigned int digest[4];
  md5_2words(key.words, 3, digest);
  for (int i = 0; i < 4; ++i) {
    if (digest[i] != MD5_ABC[i]) {
      printf("*** host reference error: md5(\"abc\")[%d]=0x%08x, expected 0x%08x\n",
             i, digest[i], MD5_ABC[i]);
      return false;
    }
  }
  return true;
}

cl_device_id device_id = NULL;
cl_context context = NULL;
cl_command_queue commandQueue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_mem found_index_memobj = NULL;
cl_mem found_key_memobj = NULL;
cl_mem found_digest_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (kernel) clReleaseKernel(kernel);
  if (program) clReleaseProgram(program);
  if (found_index_memobj) clReleaseMemObject(found_index_memobj);
  if (found_key_memobj) clReleaseMemObject(found_key_memobj);
  if (found_digest_memobj) clReleaseMemObject(found_digest_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int byte_length = 3;
int vals_per_byte = 16;

static void show_usage() {
  printf("Usage: [-l byte length] [-v values per byte] [-h: help]\n");
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "l:v:h")) != -1) {
    switch (c) {
    case 'l':
      byte_length = atoi(optarg);
      break;
    case 'v':
      vals_per_byte = atoi(optarg);
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

  printf("MD5 keyspace: byte_length=%d vals_per_byte=%d\n",
         byte_length, vals_per_byte);
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (byte_length < 1 || byte_length > 7) {
    printf("Error: byte length must be in [1, 7] (SHOC md5_2words limit)\n");
    return -1;
  }
  if (vals_per_byte < 1) {
    printf("Error: values per byte must be at least 1\n");
    return -1;
  }

  // SHOC FindKeyspaceSize (MD5Hash.cpp:188), with an int bound.
  long long keyspace_ll = 1;
  for (int i = 0; i < byte_length; ++i)
    keyspace_ll *= vals_per_byte;
  if (keyspace_ll > 0x7fffffffLL) {
    printf("Error: keyspace %lld exceeds the kernel's int range\n", keyspace_ll);
    return -1;
  }
  const int keyspace = (int)keyspace_ll;
  const int nthreads = keyspace / vals_per_byte;

  printf("Keyspace: %d keys, %d work-items x %d keys\n",
         keyspace, nthreads, vals_per_byte);

  if (!check_md5_core_against_abc())
    return -1;

  // Host reference: index 0 of the keyspace (SHOC IndexToKey + md5_2words).
  const int target_index = 0;
  union { unsigned int words[2]; unsigned char bytes[8]; } target_key;
  target_key.words[0] = 0;
  target_key.words[1] = 0;
  IndexToKey(target_index, byte_length, vals_per_byte, target_key.bytes);

  unsigned int target_digest[4];
  md5_2words(target_key.words, byte_length, target_digest);

  printf("Target index %d, key bytes:", target_index);
  for (int i = 0; i < byte_length; ++i) printf(" %02x", target_key.bytes[i]);
  printf("\nTarget digest: %08x %08x %08x %08x\n",
         target_digest[0], target_digest[1], target_digest[2], target_digest[3]);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  printf("Create program from kernel source\n");
  size_t kernel_size = 0;
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  kernel = CL_CHECK2(clCreateKernel(program, KERNEL_NAME, &_err));

  int found_index = -1;
  unsigned char found_key[8] = {0};
  unsigned int found_digest[4] = {0};

  found_index_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                sizeof(int), NULL, &_err));
  found_key_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              8 * sizeof(unsigned char), NULL, &_err));
  found_digest_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                 4 * sizeof(unsigned int), NULL, &_err));

  CL_CHECK(clSetKernelArg(kernel, 0, sizeof(unsigned int), (void *)&target_digest[0]));
  CL_CHECK(clSetKernelArg(kernel, 1, sizeof(unsigned int), (void *)&target_digest[1]));
  CL_CHECK(clSetKernelArg(kernel, 2, sizeof(unsigned int), (void *)&target_digest[2]));
  CL_CHECK(clSetKernelArg(kernel, 3, sizeof(unsigned int), (void *)&target_digest[3]));
  CL_CHECK(clSetKernelArg(kernel, 4, sizeof(int), (void *)&keyspace));
  CL_CHECK(clSetKernelArg(kernel, 5, sizeof(int), (void *)&byte_length));
  CL_CHECK(clSetKernelArg(kernel, 6, sizeof(int), (void *)&vals_per_byte));
  CL_CHECK(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&found_index_memobj));
  CL_CHECK(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&found_key_memobj));
  CL_CHECK(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&found_digest_memobj));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload output buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, found_index_memobj, CL_TRUE, 0,
                                sizeof(int), &found_index, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, found_key_memobj, CL_TRUE, 0,
                                8 * sizeof(unsigned char), found_key, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, found_digest_memobj, CL_TRUE, 0,
                                4 * sizeof(unsigned int), found_digest, 0, NULL, NULL));

  printf("Execute the kernel\n");
  size_t local_work_size = LWS;
  size_t global_work_size = ((size_t)nthreads + LWS - 1) / LWS * LWS;
  auto time_start = std::chrono::high_resolution_clock::now();
  CL_CHECK(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                                  &global_work_size, &local_work_size,
                                  0, NULL, NULL));
  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download result\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, found_index_memobj, CL_TRUE, 0,
                               sizeof(int), &found_index, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, found_key_memobj, CL_TRUE, 0,
                               8 * sizeof(unsigned char), found_key, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(commandQueue, found_digest_memobj, CL_TRUE, 0,
                               4 * sizeof(unsigned int), found_digest, 0, NULL, NULL));

  printf("Verify result\n");
  int errors = 0;
  if (found_index != target_index) {
    printf("*** error: found index=%d, expected %d\n", found_index, target_index);
    ++errors;
  }
  for (int i = 0; i < 8; ++i) {
    if (found_key[i] != target_key.bytes[i]) {
      if (errors < 100)
        printf("*** error: found key[%d]=0x%02x, expected 0x%02x\n",
               i, found_key[i], target_key.bytes[i]);
      ++errors;
    }
  }
  for (int i = 0; i < 4; ++i) {
    if (found_digest[i] != target_digest[i]) {
      if (errors < 100)
        printf("*** error: found digest[%d]=0x%08x, expected 0x%08x\n",
               i, found_digest[i], target_digest[i]);
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
