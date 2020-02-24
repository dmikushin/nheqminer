set(NAME djezo)

file(GLOB SRC_LIST djezo.cu)
file(GLOB HEADERS djezo.hpp digit_first.cuh digit_1.cuh digit_2.cuh digit_3.cuh digit_4.cuh digit_5.cuh digit_6.cuh digit_7.cuh digit_8.cuh digit_last_wdc.cuh)

FIND_PACKAGE(CUDA REQUIRED)

if (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")
set(CUDA_NVCC_FLAGS_RELEASE "-DNDEBUG;-D_FORCE_INLINES;-O3;--use_fast_math;-lineinfo;--ptxas-options=-v;-Xptxas=-dlcm=ca;-Xptxas=-dscm=cs;${CUDA_NVCC_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE_UPPER STREQUAL "DEBUG")
set(CUDA_NVCC_FLAGS_DEBUG "-G;-O0;${CUDA_NVCC_FLAGS_DEBUG}")
endif (CMAKE_BUILD_TYPE_UPPER STREQUAL "RELEASE")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_75,code=sm_75)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(..)
CUDA_ADD_LIBRARY(${NAME} STATIC ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${NAME} ${CUDA_LIBRARIES} cuda)

message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

install( TARGETS ${NAME} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${NAME} )
