#ifndef COMMON_H
#define COMMON_H

#ifndef TYPE
#define TYPE float
#endif

/* Abramowitz & Stegun 7.1.26; |error| < 1.5e-7 over the whole range. */
static inline float vx_erf(float x) {
  const float p  = 0.3275911f;
  const float a1 = 0.254829592f;
  const float a2 = -0.284496736f;
  const float a3 = 1.421413741f;
  const float a4 = -1.453152027f;
  const float a5 = 1.061405429f;

  float sign = x < 0.0f ? -1.0f : 1.0f;
  float ax = x < 0.0f ? -x : x;
  float t = 1.0f / (1.0f + p * ax);
  float poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
  return sign * (1.0f - poly * exp(-ax * ax));
}

#endif // COMMON_H
