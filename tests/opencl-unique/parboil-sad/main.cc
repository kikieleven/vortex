// Parboil sad — standalone self-checking OpenCL port for Vortex.
//
// Source:
//   third_benchmark/parboil/benchmarks/sad/src/opencl_base/sad_kernel.cl
//   third_benchmark/parboil/benchmarks/sad/src/opencl_base/main.cpp
//     (launch geometry: 61 search-position chunks per 4x4 block, then the
//      larger-block kernels at 32x4 / 32x1)
//   third_benchmark/parboil/benchmarks/sad/src/cpu/sad_cpu.c
//     (sad4_cpu / larger_sads: the host reference used here)
//
// Parboil reads two BMP frames and a golden SAD file; this port generates
// deterministic 8-bit frames and compares the full 41-row SAD array against
// the serial CPU reference.
//
// Adaptations (see kernel.cl):
//   - the reference frame is a buffer plus an explicit CLAMP_TO_EDGE read, not
//     an image2d;
//   - work-groups are 16 work-items with the logical thread indices iterated
//     inside the kernel;
//   - ushort2 stores are scalarised.
//
// Data: deterministic rand()%256 pixels in a (16*mb)x(16*mb) frame. All SAD
// values are exact integer sums, so the comparison is bit-exact.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <unistd.h>
#include <string.h>
#include <chrono>
#include <vector>

#include "common.h"

#define KERNEL_MB "mb_sad_calc"
#define KERNEL_L8 "larger_sad_calc_8"
#define KERNEL_L16 "larger_sad_calc_16"

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
cl_kernel mb_kernel = NULL;
cl_kernel l8_kernel = NULL;
cl_kernel l16_kernel = NULL;
cl_mem sads_memobj = NULL;
cl_mem frame_memobj = NULL;
cl_mem ref_memobj = NULL;
uint8_t* kernel_bin = NULL;

static void cleanup() {
  if (commandQueue) clReleaseCommandQueue(commandQueue);
  if (mb_kernel) clReleaseKernel(mb_kernel);
  if (l8_kernel) clReleaseKernel(l8_kernel);
  if (l16_kernel) clReleaseKernel(l16_kernel);
  if (program) clReleaseProgram(program);
  if (sads_memobj) clReleaseMemObject(sads_memobj);
  if (frame_memobj) clReleaseMemObject(frame_memobj);
  if (ref_memobj) clReleaseMemObject(ref_memobj);
  if (context) clReleaseContext(context);
  if (device_id) clReleaseDevice(device_id);

  if (kernel_bin) free(kernel_bin);
}

int size = 2;  // macroblocks per side

static void show_usage() {
  printf("Usage: [-n macroblocks per side] [-h: help]\n");
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

// ---- CPU reference: sad_cpu.c --------------------------------------------

static unsigned short line_buf[16];

static unsigned short* create_padded_row(const unsigned short* ref, int y, int x,
                                         int height, int width) {
  if (y < 0) y = 0;
  else if (y >= height) y = height - 1;

  const unsigned short* row = ref + y * width;

  if ((x >= 0) && (x <= width - 16)) return const_cast<unsigned short*>(row + x);

  int i = 0;
  for (; (x < 0) && (i < 16); x++, i++)
    line_buf[i] = row[0];
  for (; (x < width) && (i < 16); x++, i++)
    line_buf[i] = row[x];
  for (; i < 16; x++, i++)
    line_buf[i] = row[width - 1];

  return line_buf;
}

static void sad4_one_macroblock(unsigned short* macroblock_sad,
                                const unsigned short* frame,
                                const unsigned short* ref,
                                int frame_y, int frame_x,
                                int mb_width, int mb_height) {
  unsigned short frame_mb[256];
  int width = mb_width * 16;
  int max_width = width - 17;
  int height = mb_height * 16;
  int max_height = height - 17;
  int range_partly_outside;

  for (int y = 0; y < 16; y++)
    for (int x = 0; x < 16; x++)
      frame_mb[16*y+x] = frame[width * y + x];

  if ((frame_x >= SEARCH_RANGE) && (frame_x <= width - SEARCH_RANGE - 17) &&
      (frame_y >= SEARCH_RANGE) && (frame_y <= height - SEARCH_RANGE - 17))
    range_partly_outside = 0;
  else
    range_partly_outside = 1;

  int pos = 0;
  for (int pos_y = -SEARCH_RANGE; pos_y <= SEARCH_RANGE; pos_y++) {
    for (int pos_x = -SEARCH_RANGE; pos_x <= SEARCH_RANGE; pos_x++, pos++) {
      int abs_x = frame_x + pos_x;
      int abs_y = frame_y + pos_y;
      int do_bounds_check = range_partly_outside &&
        !(abs_y >= 0 && abs_y <= max_height &&
          abs_x >= 0 && abs_x <= max_width);

      const unsigned short* curptr = frame_mb;
      unsigned short* sad_line = macroblock_sad;

      for (int blky = 0; blky < 4; blky++) {
        int sad0 = 0, sad1 = 0, sad2 = 0, sad3 = 0;

        for (int y = 0; y < 4; y++) {
          const unsigned short* refptr = do_bounds_check
            ? create_padded_row(ref, abs_y, abs_x, height, width)
            : (ref + abs_y * width + abs_x);
          abs_y++;

          sad0 += abs((int)refptr[0] - (int)curptr[0]);
          sad0 += abs((int)refptr[1] - (int)curptr[1]);
          sad0 += abs((int)refptr[2] - (int)curptr[2]);
          sad0 += abs((int)refptr[3] - (int)curptr[3]);
          sad1 += abs((int)refptr[4] - (int)curptr[4]);
          sad1 += abs((int)refptr[5] - (int)curptr[5]);
          sad1 += abs((int)refptr[6] - (int)curptr[6]);
          sad1 += abs((int)refptr[7] - (int)curptr[7]);
          sad2 += abs((int)refptr[8] - (int)curptr[8]);
          sad2 += abs((int)refptr[9] - (int)curptr[9]);
          sad2 += abs((int)refptr[10] - (int)curptr[10]);
          sad2 += abs((int)refptr[11] - (int)curptr[11]);
          sad3 += abs((int)refptr[12] - (int)curptr[12]);
          sad3 += abs((int)refptr[13] - (int)curptr[13]);
          sad3 += abs((int)refptr[14] - (int)curptr[14]);
          sad3 += abs((int)refptr[15] - (int)curptr[15]);

          curptr += 16;
        }

        sad_line[pos] = sad0;
        sad_line[MAX_POS_PADDED + pos] = sad1;
        sad_line[MAX_POS_PADDED*2 + pos] = sad2;
        sad_line[MAX_POS_PADDED*3 + pos] = sad3;

        sad_line += MAX_POS_PADDED*4;
      }
    }
  }
}

static void sad4_cpu(unsigned short* blk_sad,
                     const unsigned short* frame,
                     const unsigned short* ref,
                     int mb_width, int mb_height) {
  blk_sad += SAD_TYPE_7_IX(mb_width * mb_height);

  unsigned int frame_yoff = 0;
  for (int mb_y = 0; mb_y < mb_height; mb_y++, frame_yoff += 256 * mb_width) {
    for (int mb_x = 0; mb_x < mb_width; mb_x++) {
      sad4_one_macroblock(blk_sad + (mb_y * mb_width + mb_x) * (SAD_TYPE_7_CT * MAX_POS_PADDED),
                          frame + frame_yoff + mb_x * 16,
                          ref,
                          mb_y * 16, mb_x * 16,
                          mb_width, mb_height);
    }
  }
}

static void add_vectors(unsigned short* z, const unsigned short* x,
                        const unsigned short* y) {
  // sad_cpu.c ADD_VECTORS: (MAX_POS+1)/2 pairs, i.e. 1090 elements.
  for (int count = 0; count < (MAX_POS+1)/2; count++, z += 2, x += 2, y += 2) {
    z[0] = x[0] + y[0];
    z[1] = x[1] + y[1];
  }
}

static void larger_sads(unsigned short* sads, int mbs) {
  for (int macroblock = 0; macroblock < mbs; macroblock++) {
    // Block type 6
    for (int block_y = 0; block_y < 2; block_y++)
      for (int block_x = 0; block_x < 4; block_x++) {
        const unsigned short* x = sads + SAD_TYPE_7_IX(mbs) +
          macroblock * SAD_TYPE_7_CT * MAX_POS_PADDED +
          (8 * block_y + block_x) * MAX_POS_PADDED;
        const unsigned short* y = x + 4 * MAX_POS_PADDED;
        unsigned short* z = sads + SAD_TYPE_6_IX(mbs) +
          macroblock * SAD_TYPE_6_CT * MAX_POS_PADDED +
          (4 * block_y + block_x) * MAX_POS_PADDED;
        add_vectors(z, x, y);
      }

    // Block type 5
    for (int block_y = 0; block_y < 4; block_y++)
      for (int block_x = 0; block_x < 2; block_x++) {
        const unsigned short* x = sads + SAD_TYPE_7_IX(mbs) +
          macroblock * SAD_TYPE_7_CT * MAX_POS_PADDED +
          (4 * block_y + 2 * block_x) * MAX_POS_PADDED;
        const unsigned short* y = x + MAX_POS_PADDED;
        unsigned short* z = sads + SAD_TYPE_5_IX(mbs) +
          macroblock * SAD_TYPE_6_CT * MAX_POS_PADDED +
          (2 * block_y + block_x) * MAX_POS_PADDED;
        add_vectors(z, x, y);
      }

    // Block type 4
    for (int block_y = 0; block_y < 2; block_y++)
      for (int block_x = 0; block_x < 2; block_x++) {
        const unsigned short* x = sads + SAD_TYPE_5_IX(mbs) +
          macroblock * SAD_TYPE_5_CT * MAX_POS_PADDED +
          (4 * block_y + block_x) * MAX_POS_PADDED;
        const unsigned short* y = x + 2 * MAX_POS_PADDED;
        unsigned short* z = sads + SAD_TYPE_4_IX(mbs) +
          macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED +
          (2 * block_y + block_x) * MAX_POS_PADDED;
        add_vectors(z, x, y);
      }

    // Block type 3
    {
      const unsigned short* x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      const unsigned short* y = x + 2 * MAX_POS_PADDED;
      unsigned short* z = sads + SAD_TYPE_3_IX(mbs) +
        macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED;
      add_vectors(z, x, y);
    }
    {
      const unsigned short* x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED + MAX_POS_PADDED;
      const unsigned short* y = x + 2 * MAX_POS_PADDED;
      unsigned short* z = sads + SAD_TYPE_3_IX(mbs) +
        macroblock * SAD_TYPE_3_CT * MAX_POS_PADDED + MAX_POS_PADDED;
      add_vectors(z, x, y);
    }

    // Block type 2
    {
      const unsigned short* x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED;
      const unsigned short* y = x + MAX_POS_PADDED;
      unsigned short* z = sads + SAD_TYPE_2_IX(mbs) +
        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      add_vectors(z, x, y);
    }
    {
      const unsigned short* x = sads + SAD_TYPE_4_IX(mbs) +
        macroblock * SAD_TYPE_4_CT * MAX_POS_PADDED + 2 * MAX_POS_PADDED;
      const unsigned short* y = x + MAX_POS_PADDED;
      unsigned short* z = sads + SAD_TYPE_2_IX(mbs) +
        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED + MAX_POS_PADDED;
      add_vectors(z, x, y);
    }

    // Block type 1
    {
      const unsigned short* x = sads + SAD_TYPE_2_IX(mbs) +
        macroblock * SAD_TYPE_2_CT * MAX_POS_PADDED;
      const unsigned short* y = x + MAX_POS_PADDED;
      unsigned short* z = sads + SAD_TYPE_1_IX(mbs) +
        macroblock * SAD_TYPE_1_CT * MAX_POS_PADDED;
      add_vectors(z, x, y);
    }
  }
}

int main(int argc, char **argv) {
  parse_args(argc, argv);

  if (size < 1) {
    printf("Error: size must be positive\n");
    return -1;
  }

  const int mb_width = size;
  const int mb_height = size;
  const int width = mb_width * 16;
  const int height = mb_height * 16;
  const int image_size_macroblocks = mb_width * mb_height;
  const size_t image_size = (size_t)width * height;
  const size_t sad_size = (size_t)41 * MAX_POS_PADDED * image_size_macroblocks;

  printf("Image size=%dx%d (%dx%d macroblocks)\n", width, height, mb_width, mb_height);

  std::vector<unsigned short> h_frame(image_size), h_ref(image_size);
  srand(1);
  for (size_t i = 0; i < image_size; ++i)
    h_frame[i] = (unsigned short)(rand() % 256);
  for (size_t i = 0; i < image_size; ++i)
    h_ref[i] = (unsigned short)(rand() % 256);
  std::vector<unsigned short> h_sads(sad_size, 0);

  cl_platform_id platform_id;
  CL_CHECK(clGetPlatformIDs(1, &platform_id, NULL));
  CL_CHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL));

  printf("Create context\n");
  context = CL_CHECK2(clCreateContext(NULL, 1, &device_id, NULL, NULL, &_err));

  char device_string[1024];
  clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_string), &device_string, NULL);
  printf("Using device: %s\n", device_string);

  printf("Allocate device buffers\n");
  sads_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_WRITE, sad_size * sizeof(unsigned short), NULL, &_err));
  frame_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, image_size * sizeof(unsigned short), NULL, &_err));
  ref_memobj = CL_CHECK2(clCreateBuffer(context, CL_MEM_READ_ONLY, image_size * sizeof(unsigned short), NULL, &_err));

  printf("Create program from kernel source\n");
  size_t kernel_size = 0;
  if (0 != read_kernel_file("kernel.cl", &kernel_bin, &kernel_size))
    return -1;
  program = CL_CHECK2(clCreateProgramWithSource(
    context, 1, (const char**)&kernel_bin, &kernel_size, &_err));
  if (program == NULL) {
    cleanup();
    return -1;
  }

  CL_CHECK(clBuildProgram(program, 1, &device_id, NULL, NULL, NULL));

  mb_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_MB, &_err));
  l8_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_L8, &_err));
  l16_kernel = CL_CHECK2(clCreateKernel(program, KERNEL_L16, &_err));

  commandQueue = CL_CHECK2(clCreateCommandQueue(context, device_id, 0, &_err));

  printf("Upload source buffers\n");
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, sads_memobj, CL_TRUE, 0, sad_size * sizeof(unsigned short), h_sads.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, frame_memobj, CL_TRUE, 0, image_size * sizeof(unsigned short), h_frame.data(), 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(commandQueue, ref_memobj, CL_TRUE, 0, image_size * sizeof(unsigned short), h_ref.data(), 0, NULL, NULL));

  printf("Execute the kernel\n");
  const int width4 = (width + 3) / 4;
  const int height4 = (height + 3) / 4;
  auto time_start = std::chrono::high_resolution_clock::now();

  CL_CHECK(clSetKernelArg(mb_kernel, 0, sizeof(cl_mem), (void *)&sads_memobj));
  CL_CHECK(clSetKernelArg(mb_kernel, 1, sizeof(cl_mem), (void *)&frame_memobj));
  CL_CHECK(clSetKernelArg(mb_kernel, 2, sizeof(int), (void *)&mb_width));
  CL_CHECK(clSetKernelArg(mb_kernel, 3, sizeof(int), (void *)&mb_height));
  CL_CHECK(clSetKernelArg(mb_kernel, 4, sizeof(cl_mem), (void *)&ref_memobj));
  CL_CHECK(clSetKernelArg(mb_kernel, 5, sizeof(int), (void *)&width));
  CL_CHECK(clSetKernelArg(mb_kernel, 6, sizeof(int), (void *)&height));
  {
    const size_t global[3] = {((CEIL_POS + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE,
                              (size_t)width4, (size_t)height4};
    const size_t local[3] = {GROUP_SIZE, 1, 1};
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, mb_kernel, 3, NULL,
                                    global, local, 0, NULL, NULL));
  }

  CL_CHECK(clSetKernelArg(l8_kernel, 0, sizeof(cl_mem), (void *)&sads_memobj));
  CL_CHECK(clSetKernelArg(l8_kernel, 1, sizeof(int), (void *)&mb_width));
  CL_CHECK(clSetKernelArg(l8_kernel, 2, sizeof(int), (void *)&mb_height));
  {
    const size_t global[3] = {(size_t)mb_width * GROUP_SIZE, (size_t)mb_height, 1};
    const size_t local[3] = {GROUP_SIZE, 1, 1};
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, l8_kernel, 3, NULL,
                                    global, local, 0, NULL, NULL));
  }

  CL_CHECK(clSetKernelArg(l16_kernel, 0, sizeof(cl_mem), (void *)&sads_memobj));
  CL_CHECK(clSetKernelArg(l16_kernel, 1, sizeof(int), (void *)&mb_width));
  CL_CHECK(clSetKernelArg(l16_kernel, 2, sizeof(int), (void *)&mb_height));
  {
    const size_t global[3] = {(size_t)mb_width * GROUP_SIZE, (size_t)mb_height, 1};
    const size_t local[3] = {GROUP_SIZE, 1, 1};
    CL_CHECK(clEnqueueNDRangeKernel(commandQueue, l16_kernel, 3, NULL,
                                    global, local, 0, NULL, NULL));
  }

  CL_CHECK(clFinish(commandQueue));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  printf("Download destination buffer\n");
  CL_CHECK(clEnqueueReadBuffer(commandQueue, sads_memobj, CL_TRUE, 0,
                               sad_size * sizeof(unsigned short), h_sads.data(),
                               0, NULL, NULL));

  printf("Verify result\n");
  std::vector<unsigned short> h_ref_sads(sad_size, 0);
  sad4_cpu(h_ref_sads.data(), h_frame.data(), h_ref.data(), mb_width, mb_height);
  larger_sads(h_ref_sads.data(), image_size_macroblocks);

  int errors = 0;
  for (size_t i = 0; i < sad_size; ++i) {
    if (h_ref_sads[i] != h_sads[i]) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%u, actual=%u\n",
               (int)i, h_ref_sads[i], h_sads[i]);
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