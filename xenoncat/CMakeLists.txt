set(NAME xenoncat)

#cpu_xenoncat/
file(GLOB SRC_LIST
    xenoncat.cpp )
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
