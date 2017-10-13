set(NAME djezo)

file(GLOB SRC_LIST djezo.cu)
file(GLOB HEADERS djezo.hpp digit_first.cuh digit_1.cuh digit_2.cuh digit_3.cuh digit_4.cuh digit_5.cuh digit_6.cuh digit_7.cuh digit_8.cuh digit_last_wdc.cuh)

FIND_PACKAGE(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-D_FORCE_INLINES;--ptxas-options=-v;-Xptxas=-dlcm=ca;-Xptxas=-dscm=cs; -O3) #; -maxrregcount=32)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(..)
CUDA_ADD_LIBRARY(${NAME} STATIC ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${NAME} ${CUDA_LIBRARIES} cuda)

message("-- CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

install( TARGETS ${NAME} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${NAME} )
