// Run executable with `OMP_NUM_THREADS=4 OMP_PLACES=threads OMP_PROC_BIND=true OMP_DISPLAY_AFFINITY=true likwid_test`
// Output should look similar:
//
//Results for thread 0 (hw thread 1):
//Event ACTUAL_CPU_CLOCK:                 422688
//Event MAX_CPU_CLOCK:            632121
//Event RETIRED_INSTRUCTIONS:             179777
//Event CPU_CLOCKS_UNHALTED:              125078
//Event RETIRED_SSE_AVX_FLOPS_SINGLE_ALL:                 1024
//Event MERGE:            0
//Results for thread 1 (hw thread 3):
//Event ACTUAL_CPU_CLOCK:                 614119
//Event MAX_CPU_CLOCK:            658182
//Event RETIRED_INSTRUCTIONS:             336703
//Event CPU_CLOCKS_UNHALTED:              220149
//Event RETIRED_SSE_AVX_FLOPS_SINGLE_ALL:                 1024
//Event MERGE:            0
//Results for thread 2 (hw thread 5):
//Event ACTUAL_CPU_CLOCK:                 517851
//Event MAX_CPU_CLOCK:            581742
//Event RETIRED_INSTRUCTIONS:             326627
//Event CPU_CLOCKS_UNHALTED:              200616
//Event RETIRED_SSE_AVX_FLOPS_SINGLE_ALL:                 1024
//Event MERGE:            0
//Results for thread 3 (hw thread 7):
//Event ACTUAL_CPU_CLOCK:                 554137
//Event MAX_CPU_CLOCK:            629097
//Event RETIRED_INSTRUCTIONS:             126078
//Event CPU_CLOCKS_UNHALTED:              189136
//Event RETIRED_SSE_AVX_FLOPS_SINGLE_ALL:                 1024


#include "sycl/sycl.hpp"
#include <string>
#include <likwid.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <chrono>

using clock_type = std::chrono::steady_clock;
using time_point_t = std::chrono::time_point<clock_type>;

class LikwidPerfctrResult {
public:
    std::vector<std::vector<double>> results;
    std::vector<std::chrono::nanoseconds> times;
    std::vector<time_point_t> lastStartTime;
    std::vector<unsigned int> callCounts;

    const int num_threads, num_events, num_metrics;

    LikwidPerfctrResult(int nthreads, int nevents, int nmetrics)
            : results(nthreads, std::vector<double>(nevents)),
              times(nthreads), lastStartTime(nthreads), callCounts(nthreads),
              num_threads(nthreads), num_events(nevents), num_metrics(nmetrics) {
    }
};

class LikwidPerftool : public hipsycl::ext::performance_tool_api {
    int groupId = -1, nthreads = -1, nevents = -1, nmetrics = -1;
    char const *event_group;
    int *hwthreads;
    LikwidPerfctrResult *results;
    std::unordered_map<std::string, LikwidPerfctrResult> resultsPerGroup;
public:
    LikwidPerftool(char const *event_group) : event_group(event_group) {
        nthreads = omp_get_max_threads();
        hwthreads = new int[nthreads];
    }

    ~LikwidPerftool() {
        std::cout << "Recorded data: \n";
        print_results();
        print_group_results();

        delete results;
        delete hwthreads;
        perfmon_finalize();
    }

    virtual void init(hipsycl::rt::device_id device_id) override {
        int nthreads = omp_get_max_threads();
#pragma omp parallel shared(hwthreads)
        {
            if (getcpu((unsigned int *) (hwthreads + omp_get_thread_num()), NULL) != 0) {
                throw std::runtime_error(
                        "Cannot get cpu id on thread " + std::to_string(omp_get_thread_num()) + ", errno: " +
                        std::to_string(errno));
            }
#pragma omp critical
            std::cout << "Thread " << omp_get_thread_num() << ": hw thread " << hwthreads[omp_get_thread_num()]
                      << ", tid: " << gettid() << "\n";
        }

        perfmon_init(nthreads, (int *) hwthreads);
        groupId = perfmon_addEventSet(event_group);
        if (perfmon_setupCounters(groupId) < 0) {
            throw std::runtime_error("Cannot setup hardware counters!");
        }
        nevents = perfmon_getNumberOfEvents(groupId);
        nmetrics = perfmon_getNumberOfMetrics(groupId);
        results = new LikwidPerfctrResult(nthreads, nevents, nmetrics);
        perfmon_startCounters();
    }

    virtual void kernel_start(std::type_info const& kernel_type_info) override {
        if (groupId >= 0) {
            resultsPerGroup.try_emplace(kernel_type_info.name(), nthreads, nevents, nmetrics);
        }
    }

    virtual void kernel_start(std::string_view kernel_type_info) override {
        if (groupId >= 0) {
            resultsPerGroup.try_emplace(std::string(kernel_type_info), nthreads, nevents, nmetrics);
        }
    }

    virtual void kernel_end(std::type_info const& kernel_type_info) override {
    }

    virtual void kernel_end(std::string_view kernel_type_info) override {
    }

    virtual void omp_thread_start(std::type_info const& kernel_type_info) override {
        resultsPerGroup.at(kernel_type_info.name()).lastStartTime[omp_get_thread_num()] = clock_type::now();
    }

    virtual void omp_thread_start(std::string_view kernel_type_info) override {
        resultsPerGroup.at(std::string(kernel_type_info)).lastStartTime[omp_get_thread_num()] = clock_type::now();
    }

    virtual void omp_thread_end(std::type_info const& kernel_type_info) override {
        omp_thread_end(kernel_type_info.name());
    }

    virtual void omp_thread_end(std::string_view kernel_type_info) override {
        int thread_num = omp_get_thread_num();
        perfmon_readCountersCpu(hwthreads[thread_num]);
        auto& r = resultsPerGroup.at(std::string(kernel_type_info));
        r.times[thread_num] +=
                clock_type::now() - r.lastStartTime[thread_num];
        r.callCounts[thread_num]++;
        for (int i = 0; i < results->num_events; i++) {
            double result = perfmon_getLastResult(groupId, i, thread_num);
            results->results[thread_num][i] += result;
            r.results[thread_num][i] += result;
        }
    }

    void print_results() {
        for (int i = 0; i < results->num_threads; i++) {
            std::cout << "Results for thread " << i << " (hw thread " << hwthreads[i] << "):\n";
            for (int j = 0; j < results->num_events; j++) {
                char *event_name = perfmon_getEventName(groupId, j);
                std::cout << "Event " << event_name << ": \t\t" << results->results[i][j] << "\n";
            }
            for (int j = 0; j < nmetrics; j++) {
                char *metric_name = perfmon_getMetricName(groupId, j);
                std::cout << "Metric " << metric_name << ": \t\t" << perfmon_getMetric(groupId, j, i) << "\n";
            }
        }
    }

    void print_group_results() {
        for (auto& grp: resultsPerGroup) {
            std::cout << "\n Results for group " << grp.first << "\n";
            for (int j = 0; j < grp.second.num_events; j++) {
                char *event_name = perfmon_getEventName(groupId, j);
                std::cout << "Event " << event_name << ": \t\t";
                for (int i = 0; i < grp.second.num_threads; i++) {
                    std::cout << grp.second.results[i][j] << "\t";
                }
                std::cout << "\n";
            }
        }
    }

    void write_marker_file(const char *filename) {
        std::ofstream file(filename);
        file << nthreads << " " << resultsPerGroup.size() << " 1\n";
        int regionId = 0;
        for (auto&& region: resultsPerGroup) {
            file << regionId << ":" << region.first << "-0\n";
            regionId++;
        }
        regionId = 0;
        for (auto&& region: resultsPerGroup) {
            for (int n = 0; n < region.second.num_threads; n++) {
                file << regionId << " 0 " << n << " " << region.second.callCounts[n]
                     << " ";
                file << region.second.times[n].count() * 1e-9 << " " << region.second.results[n].size();
                for (int j = 0; j < region.second.results[n].size(); j++) {
                    file << " " << region.second.results[n][j];
                }
                file << "\n";
            }
            regionId++;
        }
    }

    void print_marker_api_results(const char *filename) {
        perfmon_readMarkerFile(filename);
        int nregions = perfmon_getNumberOfRegions();
        for (int i = 0; i < nregions; i++) {
            std::cout << "Region " << perfmon_getTagOfRegion(i) << ":\n";
            int nmetrics = perfmon_getMetricsOfRegion(i);
            int nthreads = perfmon_getThreadsOfRegion(i);
            for (int j = 0; j < nmetrics; j++) {
                std::cout << perfmon_getMetricName(groupId, j) << ": ";
                for (int k = 0; k < nthreads; k++) {
                    std::cout << perfmon_getMetricOfRegionThread(i, j, k) << "  |  ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    }
};

class Triad;

class Triad2;

int main() {
    constexpr size_t len = 1000 * 1000 * 1000 / 4;
    constexpr size_t repetitions = 32;

    std::shared_ptr<LikwidPerftool> likwid = std::make_shared<LikwidPerftool>("FLOPS_SP");
    sycl::queue q(sycl::property_list{sycl::property::queue::AdaptiveCpp_instrumentation(likwid)});

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

    for (size_t i = 0; i < repetitions; i++) {
        q.submit([&](sycl::handler& cg) {
            auto&& aAcc = a.get_access<sycl::access_mode::read>(cg);
            auto&& bAcc = b.get_access<sycl::access_mode::read>(cg);
            auto&& cAcc = c.get_access<sycl::access_mode::read>(cg);
            auto&& resultAcc = result.get_access<sycl::access_mode::write>(cg);

            cg.parallel_for<Triad>(sycl::range<1>{len}, [=](sycl::id<1> i) {
                resultAcc[i] = aAcc[i] + bAcc[i] * cAcc[i];
            });
        });

        q.submit([&](sycl::handler& cg) {
            auto&& aAcc = a.get_access<sycl::access_mode::read>(cg);
            auto&& bAcc = b.get_access<sycl::access_mode::read>(cg);
            auto&& cAcc = c.get_access<sycl::access_mode::read>(cg);
            auto&& resultAcc = result.get_access<sycl::access_mode::write>(cg);

            cg.parallel_for<Triad2>(sycl::range<1>{len}, [=](sycl::id<1> i) {
                resultAcc[i] = aAcc[i] + bAcc[i] * cAcc[i];
            });
        });
    }

    q.wait();

    likwid->write_marker_file("likwid-results.txt");
    std::cout << "Results from Marker API:\n";
    likwid->print_marker_api_results("likwid-results.txt");
    std::cout << "\n";

    return 0;
}
