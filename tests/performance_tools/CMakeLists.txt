add_executable(likwid_test likwid.cpp)
add_sycl_to_target(TARGET likwid_test)
target_compile_definitions(likwid_test PUBLIC -DLIKWID_PERFMON)
target_link_libraries(likwid_test PUBLIC likwid)
