set(NAME xenoncat)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

enable_language(ASM_FASM)

file(GLOB SRC_LIST
    xenoncat.cpp
    asm/equihash_avx1.asm
    asm/equihash_avx2.asm
    )
file(GLOB HEADERS
    xenoncat.hpp
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(..)
ADD_LIBRARY(${NAME} STATIC ${SRC_LIST} ${HEADERS})
TARGET_LINK_LIBRARIES(${NAME} )

install( TARGETS ${NAME} RUNTIME DESTINATION bin ARCHIVE DESTINATION lib LIBRARY DESTINATION lib )
install( FILES ${HEADERS} DESTINATION include/${NAME} )
