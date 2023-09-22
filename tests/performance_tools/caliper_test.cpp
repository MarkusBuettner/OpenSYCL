// Run with
// CALI_USE_OMPT=1 CALI_CONFIG=openmp-report ./performance_tools/caliper_test
//
// Output should be similar to:
//
// Path                                             #Threads Time (thread) Time
// (total) Work %   Barrier % Time (work) Time (barrier)
// N7hipsycl4glue18kernel_~~rEEUlNS4_2idILi1EEEE_EE        8      0.004853
// 0.008049 2.366737 97.633263    0.000178       0.007355

#include "sycl/sycl.hpp"
#include <caliper/cali.h>

class CaliperPerftool : public hipsycl::ext::performance_tool_api {
public:
  void init(hipsycl::rt::device_id device_id) override {}

  virtual void kernel_start(std::type_info const &kernel_type_info) override {}
  virtual void kernel_end(std::type_info const &kernel_type_info) override {}

  virtual void
  omp_thread_start(std::type_info const &kernel_type_info) override {
    CALI_MARK_BEGIN(kernel_type_info.name());
  }
  virtual void omp_thread_end(std::type_info const &kernel_type_info) override {
    CALI_MARK_END(kernel_type_info.name());
  }
};

class Triad;

int main() {
  constexpr size_t len = 1024;

  std::shared_ptr<CaliperPerftool> likwid = std::make_shared<CaliperPerftool>();
  sycl::queue q(sycl::property_list{
      sycl::property::queue::hipSYCL_instrumentation(likwid)});

  sycl::buffer<float> a(len);
  sycl::buffer<float> b(len);
  sycl::buffer<float> c(len);
  sycl::buffer<float> result(len);

  {
    auto &&aAcc = a.get_host_access();
    auto &&bAcc = b.get_host_access();
    auto &&cAcc = c.get_host_access();

    for (size_t i = 0; i < len; i++) {
      aAcc[i] = 1.0;
      bAcc[i] = 2.0;
      cAcc[i] = -3.0;
    }
  }

  q.submit([&](sycl::handler &cg) {
    auto &&aAcc = a.get_access<sycl::access_mode::read>(cg);
    auto &&bAcc = b.get_access<sycl::access_mode::read>(cg);
    auto &&cAcc = c.get_access<sycl::access_mode::read>(cg);
    auto &&resultAcc = result.get_access<sycl::access_mode::write>(cg);

    cg.parallel_for<Triad>(sycl::range<1>{len}, [=](sycl::id<1> i) {
      resultAcc[i] = aAcc[i] + bAcc[i] * cAcc[i];
    });
  });

  q.wait();

  return 0;
}