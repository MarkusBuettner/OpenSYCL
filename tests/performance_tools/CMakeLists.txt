add_executable(likwid_test likwid_test.cpp)
add_sycl_to_target(TARGET likwid_test)
target_compile_definitions(likwid_test PUBLIC -DLIKWID_PERFMON)
target_link_libraries(likwid_test PUBLIC likwid)

add_executable(caliper_test caliper_test.cpp)
add_sycl_to_target(TARGET caliper_test)
target_link_libraries(caliper_test PUBLIC caliper)
