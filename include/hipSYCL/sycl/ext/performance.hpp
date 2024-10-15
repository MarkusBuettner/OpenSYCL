/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_SYCL_EXT_PERFORMANCE_HPP
#define HIPSYCL_SYCL_EXT_PERFORMANCE_HPP

#include <typeinfo>

namespace hipsycl::ext {

    class performance_tool_api {
    public:
        /**
         * This function is called during queue construction.
         */
        virtual void init(rt::device_id device_id) = 0;

        /**
         * This function is called once when the kernel is launched, before any parallelism starts
         * (similar to SYCL event::command_start).
         */
        virtual void kernel_start(std::type_info const& kernel_type_info) = 0;
        virtual void kernel_start(std::string_view sscp_kernel_name) = 0;

        /**
         * This function is called once when the kernel has finished executing (similar to SYCL event::command_end).
         */
        virtual void kernel_end(std::type_info const& kernel_type_info) = 0;
        virtual void kernel_end(std::string_view sscp_kernel_name) = 0;

        /**
         * This function is called once at the start of each OpenMP thread.
         */
        virtual void omp_thread_start(std::type_info const& kernel_type_info) = 0;
        virtual void omp_thread_start(std::string_view sscp_kernel_name) = 0;

        /**
         * This function is called once at the end of each OpenMP thread.
         */
        virtual void omp_thread_end(std::type_info const& kernel_type_info) = 0;
        virtual void omp_thread_end(std::string_view sscp_kernel_name) = 0;
    };

}

#endif //HIPSYCL_SYCL_EXT_PERFORMANCE_HPP
