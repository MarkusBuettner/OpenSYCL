add_executable(likwid_test likwid_test.cpp)
add_sycl_to_target(TARGET likwid_test)
target_compile_definitions(likwid_test PUBLIC -DLIKWID_PERFMON)
target_link_libraries(likwid_test PUBLIC likwid)

add_executable(caliper_test caliper_test.cpp)
add_sycl_to_target(TARGET caliper_test)
target_link_libraries(caliper_test PUBLIC caliper)

# You may need to adapt the CUDA paths here
set(CUDA_PATH /usr/local/cuda)
if(DEFINED ENV{CUDA_HOME})
    set(CUDA_PATH $ENV{CUDA_HOME})
endif()

add_executable(cupti_test cupti_test.cpp)
target_include_directories(cupti_test PRIVATE ${CUDA_PATH}/include ${CUDA_PATH}/extras/CUPTI/samples/extensions/include/profilerhost_util)
add_sycl_to_target(TARGET cupti_test)
target_link_directories(cupti_test PRIVATE ${CUDA_PATH}/extras/CUPTI/samples/extensions/src/profilerhost_util)
target_link_libraries(cupti_test PUBLIC cuda cupti nvperf_host nvperf_target profilerHostUtil)
