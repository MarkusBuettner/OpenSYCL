/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_SYCL_EXT_PERFORMANCE_GUARD_HPP
#define HIPSYCL_SYCL_EXT_PERFORMANCE_GUARD_HPP

#include <typeinfo>
#include <memory>

#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/sycl/ext/performance.hpp"

namespace hipsycl::ext {

class performance_api_guard : public hipsycl::ext::performance_tool_api {
public:
  explicit performance_api_guard(rt::dag_node *node) {
    if (node && node->get_execution_hints()
            .has_hint<rt::hints::performance_tool_api>()) {
      _performance_api = node->get_execution_hints()
                             .get_hint<rt::hints::performance_tool_api>()
                             ->get_performance_tool();
      _enable = _performance_api != nullptr;
    }
  }

  performance_api_guard(rt::dag_node *node, std::type_info&& kernel_type_info) {

  }

  virtual void init(hipsycl::rt::device_id) override {
    // init should never be called by the kernel launcher...
  }

  virtual void kernel_start(std::type_info const &kernel_type_info) override {
    if (_enable) {
      _performance_api->kernel_start(kernel_type_info);
    }
  }

  virtual void kernel_start(std::string_view sscp_kernel_name) override {
    if (_enable) {
      _performance_api->kernel_start(sscp_kernel_name);
    }
  }

  virtual void kernel_end(std::type_info const &kernel_type_info) override {
    if (_enable) {
      _performance_api->kernel_end(kernel_type_info);
    }
  }

  virtual void kernel_end(std::string_view kernel_type_info) override {
    if (_enable) {
      _performance_api->kernel_end(kernel_type_info);
    }
  }

  virtual void
  omp_thread_start(std::type_info const &kernel_type_info) override {
    if (_enable) {
      _performance_api->omp_thread_start(kernel_type_info);
    }
  }

  virtual void
  omp_thread_start(std::string_view kernel_type_info) override {
    if (_enable) {
      _performance_api->omp_thread_start(kernel_type_info);
    }
  }

  virtual void omp_thread_end(std::type_info const &kernel_type_info) override {
    if (_enable) {
      _performance_api->omp_thread_end(kernel_type_info);
    }
  }

  virtual void omp_thread_end(std::string_view kernel_type_info) override {
    if (_enable) {
      _performance_api->omp_thread_end(kernel_type_info);
    }
  }

private:
  std::shared_ptr<performance_tool_api> _performance_api = nullptr;
  bool _enable = false;
};

} // namespace hipsycl::ext

#endif // HIPSYCL_SYCL_EXT_PERFORMANCE_GUARD_HPP
