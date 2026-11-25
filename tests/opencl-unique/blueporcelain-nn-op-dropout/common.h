#ifndef COMMON_H
#define COMMON_H

#ifndef TYPE
#define TYPE float
#endif

/* Deterministic per-element RNG shared by the host reference and the device
 * kernel: the dropout mask must be reproducible on both sides to verify. */
static inline unsigned int vx_wang_hash(unsigned int s) {
  s = (s ^ 61u) ^ (s >> 16);
  s *= 9u;
  s = s ^ (s >> 4);
  s *= 0x27d4eb2du;
  s = s ^ (s >> 15);
  return s;
}

static inline unsigned int vx_random_int(unsigned int s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return s;
}

static inline float vx_random_float(unsigned int s) {
  return vx_random_int(s) * 2.3283064365387e-10f;
}

#endif // COMMON_H
