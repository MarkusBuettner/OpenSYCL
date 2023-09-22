//
// Created by buettner on 20.09.23.
//

/**
 * Various helper functions for working with the CUPTI API.
 */

#pragma once

#include <cupti_callbacks.h>
#include <cupti_profiler_target.h>
#include <cupti_events.h>

// Copied from CUPTI examples
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

namespace {
void initialize_cupti(CUdevice cuda_device_num) {
  CUpti_Profiler_Initialize_Params enableProfilingParams = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerInitialize(&enableProfilingParams));

  CUpti_Profiler_DeviceSupported_Params params = {
      CUpti_Profiler_DeviceSupported_Params_STRUCT_SIZE};
  params.cuDevice = cuda_device_num;
  CUPTI_API_CALL(cuptiProfilerDeviceSupported(&params));

  if (params.isSupported != CUPTI_PROFILER_CONFIGURATION_SUPPORTED) {
    ::std::cerr << "Unable to profile on device " << cuda_device_num
                << ::std::endl;

    if (params.architecture == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr << "\tdevice architecture is not supported" << ::std::endl;
    }

    if (params.sli == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr << "\tdevice sli configuration is not supported"
                  << ::std::endl;
    }

    if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr << "\tdevice vgpu configuration is not supported"
                  << ::std::endl;
    } else if (params.vGpu == CUPTI_PROFILER_CONFIGURATION_DISABLED) {
      ::std::cerr << "\tdevice vgpu configuration disabled profiling support"
                  << ::std::endl;
    }

    if (params.confidentialCompute ==
        CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr
          << "\tdevice confidential compute configuration is not supported"
          << ::std::endl;
    }

    if (params.cmp == CUPTI_PROFILER_CONFIGURATION_UNSUPPORTED) {
      ::std::cerr << "\tNVIDIA Crypto Mining Processors (CMP) are not supported"
                  << ::std::endl;
    }
    exit(2);
  }
}

CUptiResult cupti_event_name_to_id(CUdevice dev, char const* event_name, CUpti_EventID *id) {
  auto result = cuptiEventGetIdFromName(dev, event_name, id);
  if (cuptiEventGetIdFromName(dev, event_name, id) != CUPTI_SUCCESS) {
    std::cerr << event_name << " is not a valid event name!";
  }
  return result;
}
} // namespace