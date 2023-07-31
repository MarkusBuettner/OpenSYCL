// Run executable with `likwid-perfctr -g FLOPS_SP -C S0:0-3 -s 0xD -f -m likwid_test`
// Output should look similar:
//
//Region N7hipsycl4glue18kernel_name_traitsI5TriadZZ4mainENK3$_0clERNS_4sycl7handlerEEUlNS4_2idILi1EEEE_EE, Group 1: FLOPS_SP
//+-------------------+------------+------------+------------+------------+
//|    Region Info    | HWThread 0 | HWThread 2 | HWThread 4 | HWThread 6 |
//+-------------------+------------+------------+------------+------------+
//| RDTSC Runtime [s] |   0.000006 |   0.000107 |   0.000024 |   0.000014 |
//|     call count    |          1 |          1 |          1 |          1 |
//+-------------------+------------+------------+------------+------------+
//
//+----------------------------------+---------+------------+------------+------------+------------+
//|               Event              | Counter | HWThread 0 | HWThread 2 | HWThread 4 | HWThread 6 |
//+----------------------------------+---------+------------+------------+------------+------------+
//|         ACTUAL_CPU_CLOCK         |  FIXC1  |     239107 |     717670 |     181735 |     167102 |
//|           MAX_CPU_CLOCK          |  FIXC2  |     300783 |     407631 |     257313 |     236376 |
//|       RETIRED_INSTRUCTIONS       |   PMC0  |       9683 |     528505 |      48679 |      31201 |
//|        CPU_CLOCKS_UNHALTED       |   PMC1  |      23170 |     363937 |      41593 |      30474 |
//| RETIRED_SSE_AVX_FLOPS_SINGLE_ALL |   PMC2  |        512 |        512 |        512 |        512 |
//|               MERGE              |   PMC3  |          0 |          0 |          0 |          0 |
//+----------------------------------+---------+------------+------------+------------+------------+
//
//+---------------------------------------+---------+---------+--------+--------+-------------+
//|                 Event                 | Counter |   Sum   |   Min  |   Max  |     Avg     |
//+---------------------------------------+---------+---------+--------+--------+-------------+
//|         ACTUAL_CPU_CLOCK STAT         |  FIXC1  | 1305614 | 167102 | 717670 | 326403.5000 |
//|           MAX_CPU_CLOCK STAT          |  FIXC2  | 1202103 | 236376 | 407631 | 300525.7500 |
//|       RETIRED_INSTRUCTIONS STAT       |   PMC0  |  618068 |   9683 | 528505 |      154517 |
//|        CPU_CLOCKS_UNHALTED STAT       |   PMC1  |  459174 |  23170 | 363937 | 114793.5000 |
//| RETIRED_SSE_AVX_FLOPS_SINGLE_ALL STAT |   PMC2  |    2048 |    512 |    512 |         512 |
//|               MERGE STAT              |   PMC3  |       0 |      0 |      0 |           0 |
//+---------------------------------------+---------+---------+--------+--------+-------------+
//
//+----------------------+--------------+------------+--------------+--------------+
//|        Metric        |  HWThread 0  | HWThread 2 |  HWThread 4  |  HWThread 6  |
//+----------------------+--------------+------------+--------------+--------------+
//|  Runtime (RDTSC) [s] | 6.211720e-06 |     0.0001 | 2.354442e-05 | 1.361569e-05 |
//| Runtime unhalted [s] |       0.0001 |     0.0003 |       0.0001 |       0.0001 |
//|      Clock [MHz]     |    1666.2398 |  3690.2525 |    1480.3873 |    1481.7563 |
//|          CPI         |       2.3929 |     0.6886 |       0.8544 |       0.9767 |
//|     SP [MFLOP/s]     |      82.4248 |     4.7877 |      21.7461 |      37.6037 |
//+----------------------+--------------+------------+--------------+--------------+
//
//+---------------------------+-----------+--------------+-----------+--------------+
//|           Metric          |    Sum    |      Min     |    Max    |      Avg     |
//+---------------------------+-----------+--------------+-----------+--------------+
//|  Runtime (RDTSC) [s] STAT |    0.0001 | 6.211720e-06 |    0.0001 | 3.584296e-05 |
//| Runtime unhalted [s] STAT |    0.0006 |       0.0001 |    0.0003 |       0.0002 |
//|      Clock [MHz] STAT     | 8318.6359 |    1480.3873 | 3690.2525 |    2079.6590 |
//|          CPI STAT         |    4.9126 |       0.6886 |    2.3929 |       1.2282 |
//|     SP [MFLOP/s] STAT     |  146.5623 |       4.7877 |   82.4248 |      36.6406 |
//+---------------------------+-----------+--------------+-----------+--------------+


#include "sycl/sycl.hpp"
#include <likwid-marker.h>
#include <string>
#include <likwid.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <vector>

class LikwidPerfctrResult {
public:
    std::vector<std::vector<double>> results;
    std::vector<std::vector<double>> tmp;

    const int num_threads, num_events;

    LikwidPerfctrResult(int nthreads, int nevents) : results(nthreads, std::vector<double>(nevents)),
                                                     tmp(nthreads, std::vector<double>(nevents)),
                                                     num_threads(nthreads), num_events(nevents) {
    }
};

class LikwidPerftool : public hipsycl::ext::performance_tool_api {
    int groupId = -1;
    char const* event_group;
    int *hwthreads;
    LikwidPerfctrResult *results;
public:
    LikwidPerftool(char const* event_group) : event_group(event_group) {
        int nthreads = omp_get_max_threads();
        std::cout << nthreads << std::endl;
        hwthreads = new int[nthreads];
    }

    ~LikwidPerftool() {
        delete results;
        delete hwthreads;
        perfmon_finalize();
    }

    virtual void init() override {
        int nthreads = omp_get_max_threads();
#pragma omp parallel shared(hwthreads)
        {
            if (getcpu((unsigned int*) (hwthreads + omp_get_thread_num()), NULL) != 0)
            {
                throw std::runtime_error("Cannot get cpu id on thread " + std::to_string(omp_get_thread_num()) + ", errno: " + std::to_string(errno));
            }
#pragma omp critical
            std::cout << "Thread " << omp_get_thread_num() << ": hw thread " << hwthreads[omp_get_thread_num()] << ", tid: " << gettid() << "\n";
        }

        perfmon_init(nthreads, (int *) hwthreads);
        groupId = perfmon_addEventSet(event_group);
        if (perfmon_setupCounters(groupId) < 0)
        {
            throw std::runtime_error("Cannot setup hardware counters!");
        }
        int nevents = perfmon_getNumberOfEvents(groupId);
        results = new LikwidPerfctrResult(nthreads, nevents);
        perfmon_startCounters();
    }

    virtual void kernel_start(std::type_info const& kernel_type_info) override {
    }
    virtual void kernel_end(std::type_info const& kernel_type_info) override {
    }

    virtual void omp_thread_start(std::type_info const& kernel_type_info) override {
        int thread_num = omp_get_thread_num();
#pragma omp critical
        { std::cout << "start thread " << thread_num << ", hw thread: " << sched_getcpu() << ", tid: " << gettid() << "\n"; }
        perfmon_readCountersCpu(hwthreads[thread_num]);
        for (int i = 0; i < results->num_events; i++) {
            results->tmp[thread_num][i] = perfmon_getLastResult(groupId, i, thread_num);
        }
    }
    virtual void omp_thread_end(std::type_info const& kernel_type_info) override {
        int thread_num = omp_get_thread_num();
#pragma omp critical
        { std::cout << "stop thread " << thread_num << ", hw thread: " << sched_getcpu() << ", tid: " << gettid() << "\n"; }
        perfmon_readCountersCpu(hwthreads[thread_num]);
        for (int i = 0; i < results->num_events; i++) {
            results->results[thread_num][i] += perfmon_getLastResult(groupId, i, thread_num);
        }
    }

    void print_results() {
        for (int i = 0; i < results->num_threads; i++) {
            std::cout << "Results for thread " << i << " (hw thread " << hwthreads[i] << ":\n";
            for (int j = 0; j < results->num_events; j++) {
                char *event_name = perfmon_getEventName(groupId, j);
                std::cout << "Event " << event_name << ": \t\t" << results->results[i][j] << "\n";
            }
        }
    }
};

class Triad;

int main()
{
    constexpr size_t len = 1024;

    // LIKWID_MARKER_INIT;

    std::shared_ptr<LikwidPerftool> likwid = std::make_shared<LikwidPerftool>("FLOPS_SP");
    sycl::queue q(sycl::property_list{sycl::property::queue::hipSYCL_instrumentation(likwid)});

    sycl::buffer<float> a(len);
    sycl::buffer<float> b(len);
    sycl::buffer<float> c(len);
    sycl::buffer<float> result(len);

    {
        auto&& aAcc = a.get_host_access();
        auto&& bAcc = b.get_host_access();
        auto&& cAcc = c.get_host_access();

        for (size_t i = 0; i < len; i++) {
            aAcc[i] = 1.0;
            bAcc[i] = 2.0;
            cAcc[i] = -3.0;
        }
    }

    q.submit([&](sycl::handler& cg) {
        auto&& aAcc = a.get_access<sycl::access_mode::read>(cg);
        auto&& bAcc = b.get_access<sycl::access_mode::read>(cg);
        auto&& cAcc = c.get_access<sycl::access_mode::read>(cg);
        auto&& resultAcc = result.get_access<sycl::access_mode::write>(cg);

        cg.parallel_for<Triad>(sycl::range<1>{len}, [=](sycl::id<1> i) {
            resultAcc[i] = aAcc[i] + bAcc[i] * cAcc[i];
        });
    });

    q.wait();
    q.submit([&](sycl::handler& cg) {
        auto&& aAcc = a.get_access<sycl::access_mode::read>(cg);
        auto&& bAcc = b.get_access<sycl::access_mode::read>(cg);
        auto&& cAcc = c.get_access<sycl::access_mode::read>(cg);
        auto&& resultAcc = result.get_access<sycl::access_mode::write>(cg);

        cg.parallel_for<Triad>(sycl::range<1>{len}, [=](sycl::id<1> i) {
            resultAcc[i] = aAcc[i] + bAcc[i] * cAcc[i];
        });
    });

    q.wait();

    likwid->print_results();
    // LIKWID_MARKER_CLOSE;

    return 0;
}