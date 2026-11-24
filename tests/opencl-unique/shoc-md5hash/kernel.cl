// MD5 key search — ported from SHOC level1/md5hash/md5.cl.
//
// The MD5 core (macros, md5_2words, IndexToKey) lives in md5_core.h and is
// shared with the host reference in main.cc. The kernel body below is the
// SHOC FindKeyWithDigest_Kernel unchanged except for the key buffer:
// SHOC casts `unsigned char key[8]` to `unsigned int*`, which assumes
// 4-byte alignment the OpenCL C private-memory allocator does not guarantee.
// A union makes the two 32-bit words the primary object instead.

#include "md5_core.h"

__kernel void
FindKeyWithDigest_Kernel(unsigned int searchDigest0,
                         unsigned int searchDigest1,
                         unsigned int searchDigest2,
                         unsigned int searchDigest3,
                         int keyspace,
                         int byteLength, int valsPerByte,
                         __global int *foundIndex,
                         __global unsigned char *foundKey,
                         __global unsigned int *foundDigest)
{
    int threadid = (get_group_id(0)*get_local_size(0)) + get_local_id(0);

    int startindex = threadid * valsPerByte;
    union { unsigned int words[2]; unsigned char bytes[8]; } key;
    key.words[0] = 0;
    key.words[1] = 0;
    IndexToKey(startindex, byteLength, valsPerByte, key.bytes);

    for (int j=0; j < valsPerByte && startindex+j < keyspace; ++j)
    {
        unsigned int digest[4];
        md5_2words(key.words, byteLength, digest);
        if (digest[0] == searchDigest0 &&
            digest[1] == searchDigest1 &&
            digest[2] == searchDigest2 &&
            digest[3] == searchDigest3)
        {
            *foundIndex = startindex + j;
            foundKey[0] = key.bytes[0];
            foundKey[1] = key.bytes[1];
            foundKey[2] = key.bytes[2];
            foundKey[3] = key.bytes[3];
            foundKey[4] = key.bytes[4];
            foundKey[5] = key.bytes[5];
            foundKey[6] = key.bytes[6];
            foundKey[7] = key.bytes[7];
            foundDigest[0] = digest[0];
            foundDigest[1] = digest[1];
            foundDigest[2] = digest[2];
            foundDigest[3] = digest[3];
        }
        ++key.bytes[0];
    }   
}

