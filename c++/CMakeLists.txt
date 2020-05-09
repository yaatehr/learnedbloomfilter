#CMAKE CONFIG OPTIONS
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(learned_b VERSION 1.0)


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(warnings "-Wall -Wextra -D_GLIBCXX_USE_CXX11_ABI=0")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(warnings "/W4 /WX /EHsc")
endif()
if (NOT CONFIGURED_ONCE)
    set(CMAKE_CXX_FLAGS "${warnings} -g"
        CACHE STRING "Flags used by the compiler during all build types." FORCE)
    set(CMAKE_C_FLAGS   "${warnings}"
        CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_BUILD_TYPE Release)
set(Z3_DEPENDENT_LIBS "")
# message("cmake compiler id is: ${CMAKE_CXX_COMPILER_ID}" )

#BEGIN PROJECT CONFIG
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/Filters  
  )
# add_subdirectory(Filters)
# add_executable(learned_b rand_learned_benchmark.cpp)
add_executable(learned_b tau_benchmark.cpp)

# add_executable(learned_b Filters/learned_bloom.cpp)
# add_executable(learned_b parse_example.cpp)

### DEPENDENCIES 

# set(benchmark_ROOT_DIR "" CACHE PATH "/Users/yaatehr/Programs/benchmark")
# find_path(benchmark_INCLUDE_DIR "benchmark/benchmark.h"
#   PATHS ${benchmark_ROOT_DIR}
#   PATH_SUFFIXES include
#   NO_DEFAULT_PATH)
# find_path(benchmark_INCLUDE_DIR "benchmark/benchmark.h")
# find_library(benchmark_LIBRARY NAMES "benchmark"
#   PATHS ${benchmark_ROOT_DIR}
#   PATH_SUFFIXES lib lib64
#   NO_DEFAULT_PATH)
# find_library(benchmark_LIBRARY NAMES "benchmark")
# include(FindPackageHandleStandardArgs)
# find_package_handle_standard_args(benchmark FOUND_VAR benchmark_FOUND
#   REQUIRED_VARS benchmark_LIBRARY
#   benchmark_INCLUDE_DIR)
# if(benchmark_FOUND)
#   set(benchmark_LIBRARIES ${benchmark_LIBRARY})
#   set(benchmark_INCLUDE_DIRS ${benchmark_INCLUDE_DIR})
# endif()
# mark_as_advanced(benchmark_INCLUDE_DIR benchmark_LIBRARY)


list(APPEND CMAKE_PREFIX_PATH "/Users/yaatehr/programs/libtorch")
find_package(Torch REQUIRED)


### LINK LIBRARIES

#BENCHMARK
# target_include_directories(learned_b PUBLIC "${benchmark_INCLUDE_DIRS}")
# target_link_libraries(learned_b "${benchmark_LIBRARIES}")
# set_property(TARGET benchmark_LIBRARIES PROPERTY CMAKE_BUILD_TYPE Release)



if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  # Try to detect if it is necessary to link against librt.
  # Note that glibc < 2.17 required librt to be linked to use clock_gettime()
  # and friends.
  # set(CLOCK_GETTIME_REQUIRES_LIBRT_TEST_CODE
  #   "
  #   #include <time.h>
  #   int main() {
  #     timespec res;
  #     int result = clock_gettime(CLOCK_REALTIME, &res);
  #     return result == 0;
  #   }
  #   "
  # )
  # check_cxx_source_compiles(
  #   "${CLOCK_GETTIME_REQUIRES_LIBRT_TEST_CODE}"
  #   CLOCK_GETTIME_NO_REQUIRE_LIBRT
  # )
  # if (NOT CLOCK_GETTIME_NO_REQUIRE_LIBRT)
    # Try again with librt
    # message(STATUS "Failed to link against clock_gettime(), trying with librt")
    # set(CMAKE_REQUIRED_LIBRARIES_OLD "${CMAKE_REQUIRED_LIBRARIES}")
    # set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES} rt")
    # check_cxx_source_compiles(
    #   "${CLOCK_GETTIME_REQUIRES_LIBRT_TEST_CODE}"
    #   CLOCK_GETTIME_REQUIRES_LIBRT
    # )
    # set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES_OLD}")
    # if (CLOCK_GETTIME_REQUIRES_LIBRT)
  list(APPEND Z3_DEPENDENT_LIBS "rt")
  #   else()
  #     message(FATAL_ERROR "Failed to link against clock_gettime()")
  #   endif()
  # endif()
endif()


#THREADS
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
list(APPEND Z3_DEPENDENT_LIBS ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(learned_b Threads::Threads)



#TORCH
target_link_libraries(learned_b "${TORCH_LIBRARIES}")

# target_include_directories(learned_b PUBLIC "${TORCH_LIBRARIES}")

#MY DEPS
# target_link_libraries(learned_b learnedBloom)
# target_link_libraries(learned_b myUtils)
# include_directories(${PROJECT_SOURCE_DIR}/Filters/OpenBloom)


# SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS}")
set_property(TARGET learned_b PROPERTY CXX_STANDARD 14)

set(CONFIGURED_ONCE TRUE CACHE INTERNAL
    "A flag showing that CMake has configured at least once.")

# function(dump_cmake_variables)
#     get_cmake_property(_variableNames VARIABLES)
#     list (SORT _variableNames)
#     foreach (_variableName ${_variableNames})
#         if (ARGV0)
#             unset(MATCHED)
#             string(REGEX MATCH ${ARGV0} MATCHED ${_variableName})
#             if (NOT MATCHED)
#                 continue()
#             endif()
#         endif()
#         message(STATUS "${_variableName}=${${_variableName}}")
#     endforeach()
# endfunction()
# dump_cmake_variables()
# execute_process(COMMAND "${CMAKE_COMMAND}" "-E" "environment")
