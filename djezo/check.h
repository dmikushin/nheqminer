#ifndef CHECK_H
#define CHECK_H

#include <cstdlib>
#ifndef __USE_POSIX
#define __USE_POSIX // HOST_NAME_MAX
#endif
#include <limits.h>
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#define CUDA_ERR_CHECK(x)                                        \
    do { cudaError_t err = x; if (err != cudaSuccess) {          \
        char hostname[HOST_NAME_MAX] = "";                       \
        gethostname(hostname, HOST_NAME_MAX);                    \
        fprintf(stderr, "CUDA error %d \"%s\" on %s at %s:%d\n", \
            (int)err, cudaGetErrorString(err), hostname,         \
            __FILE__, __LINE__);                                 \
        if (!getenv("FREEZE_ON_ERROR")) {                        \
            fprintf(stderr, "You may want to set "               \
                "FREEZE_ON_ERROR environment "                   \
                "variable to debug the case\n");                 \
            abort();                                             \
        }                                                        \
        else {                                                   \
            fprintf(stderr, "thread 0x%zx of pid %d @ %s "       \
               "is entering infinite loop\n",                    \
               (size_t)pthread_self(), (int)getpid(), hostname); \
            while (1) usleep(1000000); /* 1 sec */               \
        }                                                        \
    }} while (0)

#endif // CHECK_H

