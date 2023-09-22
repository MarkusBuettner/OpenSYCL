/**
 * Sample code to use CUPTI for CUDA profiling. The code has been adapted from
 * the "callback_profiling" CUPTI example, usually located in
 * $(CUDA_HOME)/extras/CUPTI/samples.
 *
 * It needs the CUPTI examples in the CUDA installation, do compile the
 * callback_profiling example from there first (or at least the extensions
 * under extras/CUPTI/samples/extensions/src/profilerhost_util.
 *
 * If the same kernel is invoked multiple times, the values are averaged.
 */
#include <typeindex>
#include <unordered_map>

#include "sycl/sycl.hpp"

#include "cupti_helper.h"
#include <Eval.h>
#include <Metric.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <cupti_target.h>
#include <cxxabi.h>
#include <nvperf_host.h>

#define NVPW_API_CALL(apiFuncCall)                                             \
  do {                                                                         \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",     \
              __FILE__, __LINE__, #apiFuncCall, _status);                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUPTI_API_CALL(apiFuncCall)                                            \
  do {                                                                         \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
      const char *errstr;                                                      \
      cuptiGetResultString(_status, &errstr);                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",     \
              __FILE__, __LINE__, #apiFuncCall, errstr);                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

void createCounterDataImage(int numRanges,
                            std::vector<uint8_t> &counterDataImagePrefix,
                            std::vector<uint8_t> &counterDataScratchBuffer,
                            std::vector<uint8_t> &counterDataImage) {
  CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
  counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
  counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
  counterDataImageOptions.maxNumRanges = numRanges;
  counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
  counterDataImageOptions.maxRangeNameLength = 64;

  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
      CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};
  calculateSizeParams.pOptions = &counterDataImageOptions;
  calculateSizeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  CUPTI_API_CALL(
      cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

  CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
      CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  initializeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  initializeParams.pOptions = &counterDataImageOptions;
  initializeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;
  counterDataImage.resize(calculateSizeParams.counterDataImageSize);
  initializeParams.pCounterDataImage = &counterDataImage[0];
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
      scratchBufferSizeParams = {
          CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  scratchBufferSizeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;
  scratchBufferSizeParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(
      &scratchBufferSizeParams));
  counterDataScratchBuffer.resize(
      scratchBufferSizeParams.counterDataScratchBufferSize);

  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
      initScratchBufferParams = {
          CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
  initScratchBufferParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;
  initScratchBufferParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  initScratchBufferParams.counterDataScratchBufferSize =
      scratchBufferSizeParams.counterDataScratchBufferSize;
  initScratchBufferParams.pCounterDataScratchBuffer =
      &counterDataScratchBuffer[0];
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(
      &initScratchBufferParams));
}

struct ProfilingData_t {
  int numRanges = 2;
  bool bProfiling = false;
  std::string chipName;
  std::vector<std::string> metricNames;
  std::string CounterDataFileName = "SimpleCupti.counterdata";
  std::string CounterDataSBFileName = "SimpleCupti.counterdataSB";
  CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;
  CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay;
  bool allPassesSubmitted = true;
  std::vector<uint8_t> counterDataImagePrefix;
  std::vector<uint8_t> configImage;
  std::vector<uint8_t> counterDataImage;
  std::vector<uint8_t> counterDataScratchBuffer;

  explicit ProfilingData_t(const char *chipName) : chipName(chipName) {
    metricNames.emplace_back("dram__bytes.sum.per_second");
    metricNames.emplace_back("gpu__time_duration.sum");
    metricNames.emplace_back("sm__inst_executed.sum");
    metricNames.emplace_back(
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum");
    metricNames.emplace_back(
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum");
    metricNames.emplace_back(
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum");

    setupProfiling();
  }

  void setupProfiling() {
    /* Generate configuration for metrics, this can also be done offline*/
    NVPW_InitializeHost_Params initializeHostParams = {
        NVPW_InitializeHost_Params_STRUCT_SIZE};
    NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));

    if (!metricNames.empty()) {
      if (!NV::Metric::Config::GetConfigImage(chipName, metricNames,
                                              configImage)) {
        std::cout << "Failed to create configImage" << std::endl;
        exit(EXIT_FAILURE);
      }
      if (!NV::Metric::Config::GetCounterDataPrefixImage(
              chipName, metricNames, counterDataImagePrefix)) {
        std::cout << "Failed to create counterDataImagePrefix" << std::endl;
        exit(EXIT_FAILURE);
      }
    } else {
      std::cout << "No metrics provided to profile" << std::endl;
      exit(EXIT_FAILURE);
    }

    createCounterDataImage(numRanges, counterDataImagePrefix,
                           counterDataScratchBuffer, counterDataImage);
  }
};

void beginSession(ProfilingData_t *pProfilingData) {
  CUpti_Profiler_BeginSession_Params beginSessionParams = {
      CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
  beginSessionParams.ctx = NULL;
  beginSessionParams.counterDataImageSize =
      pProfilingData->counterDataImage.size();
  beginSessionParams.pCounterDataImage = &pProfilingData->counterDataImage[0];
  beginSessionParams.counterDataScratchBufferSize =
      pProfilingData->counterDataScratchBuffer.size();
  beginSessionParams.pCounterDataScratchBuffer =
      &pProfilingData->counterDataScratchBuffer[0];
  beginSessionParams.range = pProfilingData->profilerRange;
  beginSessionParams.replayMode = pProfilingData->profilerReplayMode;
  beginSessionParams.maxRangesPerPass = pProfilingData->numRanges;
  beginSessionParams.maxLaunchesPerPass = pProfilingData->numRanges;
  CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));
}

void setConfig(ProfilingData_t *pProfilingData) {
  CUpti_Profiler_SetConfig_Params setConfigParams = {
      CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
  setConfigParams.pConfig = &pProfilingData->configImage[0];
  setConfigParams.configSize = pProfilingData->configImage.size();
  setConfigParams.passIndex = 0;
  CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
}

void enableProfiling(ProfilingData_t *pProfilingData) {
  beginSession(pProfilingData);
  setConfig(pProfilingData);
  CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
      CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};
  if (pProfilingData->profilerReplayMode == CUPTI_KernelReplay) {
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
  } else if (pProfilingData->profilerReplayMode == CUPTI_UserReplay) {
    CUpti_Profiler_BeginPass_Params beginPassParams = {
        CUpti_Profiler_BeginPass_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerBeginPass(&beginPassParams));
    CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
  }
}

void disableProfiling(ProfilingData_t *pProfilingData) {
  CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
      CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

  if (pProfilingData->profilerReplayMode == CUPTI_UserReplay) {
    CUpti_Profiler_EndPass_Params endPassParams = {
        CUpti_Profiler_EndPass_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerEndPass(&endPassParams));
    pProfilingData->allPassesSubmitted =
        (endPassParams.allPassesSubmitted == 1) ? true : false;
  } else if (pProfilingData->profilerReplayMode == CUPTI_KernelReplay) {
    pProfilingData->allPassesSubmitted = true;
  }

  if (pProfilingData->allPassesSubmitted) {
    CUpti_Profiler_FlushCounterData_Params flushCounterDataParams = {
        CUpti_Profiler_FlushCounterData_Params_STRUCT_SIZE};
    CUPTI_API_CALL(cuptiProfilerFlushCounterData(&flushCounterDataParams));
  }

  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
      CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
  CUpti_Profiler_EndSession_Params endSessionParams = {
      CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));
}

void stopProfiling() {
  CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
      CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};

  CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));
}

void callbackHandler(void *userdata, CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid, void *cbdata) {
  ProfilingData_t *profilingData = (ProfilingData_t *)(userdata);
  const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *)cbdata;
  if (domain == CUPTI_CB_DOMAIN_DRIVER_API &&
      cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      std::cout << "CUPTI Callback: Launch kernel\n";
      enableProfiling(profilingData);
    } else {
      std::cout << "CUPTI Callback: Finish kernel\n";
      disableProfiling(profilingData);
    }
  }
}

class CuptiInterface : public hipsycl::ext::performance_tool_api {
private:
  // ProfilingData_t profilingData;
  std::unordered_map<std::type_index, ProfilingData_t> profilingDataPerKernel;
  CUpti_SubscriberHandle currentSubscription = nullptr;
  CUpti_Device_GetChipName_Params getChipNameParams = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};

public:
  ~CuptiInterface() {
    stopProfiling();
    for (auto&& data : profilingDataPerKernel) {
      int status;
      char *ret = abi::__cxa_demangle(data.first.name(), nullptr, nullptr, &status);
      std::cout << "\nKernel " << ret << ":";
      if (ret) free(ret);
      NV::Metric::Eval::PrintMetricValues(data.second.chipName,
                                          data.second.counterDataImage,
                                          data.second.metricNames);
    }
  }

  void init(hipsycl::rt::device_id device_id) override {
    HIPSYCL_DEBUG_INFO << "Init cupti interface" << std::endl;
    if (device_id.get_backend() == hipsycl::rt::backend_id::cuda) {
      int deviceNum = device_id.get_id();
      initialize_cupti(deviceNum);

      getChipNameParams.deviceIndex = deviceNum;
      CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
    } else {
      std::cout << "Not running on a CUDA device\n";
    }
  }
  void kernel_start(const std::type_info &kernel_type_info) override {
    std::cout << "SYCL: Kernel start\n";
    std::type_index idx = kernel_type_info;
    auto data = profilingDataPerKernel.find(idx);
    ProfilingData_t* profilingData;
    if (data == profilingDataPerKernel.end()) {
      auto inserted = profilingDataPerKernel.emplace(idx, ProfilingData_t(getChipNameParams.pChipName));
      profilingData = &inserted.first->second;
    } else {
      profilingData = &data->second;
    }

    assert(profilingData != nullptr);
    assert(currentSubscription == nullptr);

    CUPTI_API_CALL(cuptiSubscribe(&currentSubscription,
                                  (CUpti_CallbackFunc)callbackHandler,
                                  profilingData));
    CUPTI_API_CALL(cuptiEnableCallback(1, currentSubscription,
                                       CUPTI_CB_DOMAIN_DRIVER_API,
                                       CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
  }
  void kernel_end(const std::type_info &kernel_type_info) override {
    std::cout << "SYCL: Kernel end\n";
    CUPTI_API_CALL(cuptiUnsubscribe(currentSubscription));
    currentSubscription = nullptr;
  }
  void omp_thread_start(const std::type_info &kernel_type_info) override {}
  void omp_thread_end(const std::type_info &kernel_type_info) override {}
};

int main() {

  std::shared_ptr<CuptiInterface> cupti = std::make_shared<CuptiInterface>();
  sycl::queue q{sycl::gpu_selector_v,
                sycl::property_list{
                    sycl::property::queue::hipSYCL_instrumentation(cupti)}};

  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  sycl::buffer<float, 1> a(4096);
  q.submit([&](auto &&hdl) {
    auto aAcc = a.get_access(hdl);
    hdl.template parallel_for<class Init>(
        sycl::range<1>{4096},
        [=](sycl::id<1> id) { aAcc[id] = float(id.get(0)) + 1.0f; });
  });

  q.submit([&](auto &&cgh) {
    auto aAcc = a.get_access(cgh);
    cgh.template parallel_for<class FMA>(
        sycl::range<1>{4096},
        [=](sycl::id<1> id) { aAcc[id] = aAcc[id] * 5.0f - 2.0f; });
  });

  q.submit([&](auto &&cgh) {
    auto aAcc = a.get_access(cgh);
    cgh.template parallel_for<class FMUL>(
        sycl::range<1>{4096},
        [=](sycl::id<1> id) { aAcc[id] = aAcc[id] * 3.14f; });
  });

  std::cout << a.get_host_access()[0] << "\n";
}
