cmake_minimum_required(VERSION 3.16)
project(knn_openmp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # helps clangd / editors

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP REQUIRED)

# ---- options used by the experiments ---------------------------------------
option(KNN_NATIVE     "Compile with -march=native (AVX2 on i7-10750H)" ON)
option(KNN_NO_VEC     "Disable auto-vectorization (for with/without SIMD comparison)" OFF)
option(KNN_VEC_REPORT "Print GCC vectorization report (-fopt-info-vec)" OFF)
option(KNN_TSAN       "Build with ThreadSanitizer (debug data races)" OFF)

add_library(knn_core
  src/dataset.cpp
  src/knn_seq.cpp
  src/knn_omp.cpp
  src/knn_avx2.cpp
  src/variants.cpp
)
target_include_directories(knn_core PUBLIC include)
target_link_libraries(knn_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(knn_core PUBLIC -Wall -Wextra $<$<CONFIG:Release>:-O3>)

# Hand-written intrinsics need the instruction sets enabled even when
# KNN_NATIVE is off (the i7-10750H has AVX2 and FMA).
set_source_files_properties(src/knn_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")

# NB: never add -ffast-math globally: it changes the reference results.
if(KNN_NATIVE)
  target_compile_options(knn_core PUBLIC -march=native)
endif()
if(KNN_NO_VEC)
  # Disables auto-vectorization. Loops with `#pragma omp simd` may still be
  # vectorized: check the report (KNN_VEC_REPORT) to see what really happens.
  target_compile_options(knn_core PUBLIC -fno-tree-vectorize)
endif()
if(KNN_VEC_REPORT)
  target_compile_options(knn_core PRIVATE -fopt-info-vec-optimized -fopt-info-vec-missed)
endif()
if(KNN_TSAN)
  # libgomp is not TSan-instrumented: expect some false positives on OpenMP
  # internals. Use on small inputs only.
  target_compile_options(knn_core PUBLIC -fsanitize=thread -g -O1)
  target_link_options(knn_core PUBLIC -fsanitize=thread)
endif()

add_executable(knn src/main.cpp)
target_link_libraries(knn PRIVATE knn_core)

# Measurement helper (clock frequency under load, DRAM read bandwidth): the
# numbers the performance model needs and WSL2 does not expose.
add_executable(hwprobe tools/hwprobe.cpp)
target_link_libraries(hwprobe PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(hwprobe PRIVATE -O2 -mavx2 -mfma -Wall -Wextra)

# Stand-alone experiment: synchronisation strategies on the confusion matrix
# (critical / atomic / reduction / private counters with and without padding).
# -O2: enough to keep the loops tight without changing what is being compared.
add_executable(sync_strategies tools/sync_strategies.cpp)
target_link_libraries(sync_strategies PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(sync_strategies PRIVATE -O2 -Wall -Wextra)

# ---- tests --------------------------------------------------------------------
enable_testing()
add_test(NAME smoke
         COMMAND bash ${CMAKE_SOURCE_DIR}/tests/smoke_test.sh $<TARGET_FILE:knn>)
# every strategy must produce the same matrix as the serial loop (small N, 4 threads)
add_test(NAME sync_strategies COMMAND sync_strategies 200000 1)
set_tests_properties(sync_strategies PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=4")
