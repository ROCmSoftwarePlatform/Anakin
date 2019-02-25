/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
/*
 * handleocl.cpp
 *
 *  Created on: Sep 5, 2019
 *      Author: junpeng
 */

#include <miopen/handle.hpp>
#include <miopen/ocldeviceinfo.hpp>
#include <miopen/device_name.hpp>

namespace miopen {

void* default_allocator(void* context, size_t sz)
{
    assert(context != nullptr);
    cl_int status = CL_SUCCESS;
    auto result   = clCreateBuffer(
        reinterpret_cast<cl_context>(context), CL_MEM_READ_ONLY, sz, nullptr, &status);
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "OpenCL error creating buffer: " + std::to_string(sz));
    }
    return result;
}

void default_deallocator(void*, void* mem) { clReleaseMemObject(DataCast(mem)); }

Handle::Handle(/*cl_context _clContext, cl_device_id _device*/) /*: clContext(_clContext), device(_device)*/
{
    //cl_int status = 0;
    //cmQ = clCreateCommandQueue(clContext, device, CL_QUEUE_PROFILING_ENABLE, &status);

    allocator.allocator = default_allocator;
    allocator.deallocator = default_deallocator;
    allocator.context = clContext;


}

Handle::~Handle()
{
//    clReleaseCommandQueue(cmQ);
}

int Handle::setClEnv(cl_context _clContext, cl_device_id _device)
{
    cl_int status = 0;
    clContext = _clContext;
    device = _device;

    cmQ = clCreateCommandQueue(clContext, device, CL_QUEUE_PROFILING_ENABLE, &status);
    return status;
}

int Handle::clearClEnv()
{
    cl_int status = 0;
    if(cmQ) status = clReleaseCommandQueue(cmQ);
    return status;
}

cl_command_queue Handle::GetStream() const
{
    return cmQ;
}

void Handle::Finish() const { clFinish(cmQ); }

std::size_t Handle::GetLocalMemorySize()
{
    return miopen::GetDeviceInfo<CL_DEVICE_LOCAL_MEM_SIZE>(miopen::GetDevice(this->GetStream()));
}

std::string Handle::GetDeviceName()
{
    std::string name = miopen::GetDeviceInfo<CL_DEVICE_NAME>(miopen::GetDevice(this->GetStream()));
    return GetDeviceNameFromMap(name);
}

std::size_t Handle::GetMaxComputeUnits()
{
    return miopen::GetDeviceInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(miopen::GetDevice(this->GetStream()));
}

KernelInvoke Handle::AddKernel(const std::string& algorithm,
                               const std::string& network_config,
                               const std::string& program_name,
                               const std::string& kernel_name,
                               const std::vector<size_t>& vld,
                               const std::vector<size_t>& vgd,
                               const std::string& params,
                               std::size_t cache_index)
{
    bool is_kernel_str = algorithm.find("GEMM") != std::string::npos;
    if (!is_kernel_str)
        std::cout <<"addKernel:"<<program_name<<" kernel name:"<<kernel_name<<" params:"<<params<<std::endl;
    else
        std::cout <<"addKernel kernel name:"<<kernel_name<<" params:"<<params<<std::endl;
    auto p = miopen::LoadProgram(miopen::GetContext(this->GetStream()),
                                         miopen::GetDevice(this->GetStream()),
                                         program_name,
                                         params,
                                         is_kernel_str);
    Kernel kernel{std::move(p), kernel_name, vld, vgd};
    return this->Run(kernel);
}

KernelInvoke Handle::Run(Kernel k)
{
    auto q = this->GetStream();
    return k.Invoke(q,
                    std::bind(&Handle::SetProfilingResult,
                              std::ref(*this),
                              std::placeholders::_1));
}

void Handle::SetProfilingResult(cl_event& e)
{
    if(this->enable_profiling)
    {
        size_t st, end;
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(size_t), &st, nullptr);
        clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(size_t), &end, nullptr);
        profiling_result = ((end - st) * 1e-6);
    }
}

void Handle::EnableProfiling(bool enable)
{
    enable_profiling = enable;
}

bool Handle::IsProfilingEnabled() const 
{ 
    return enable_profiling; 
}

float Handle::GetKernelTime() const { return this->profiling_result; }

Allocator::ManageDataPtr Handle::Create(std::size_t sz)
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    return this->allocator(sz);
}

Allocator::ManageDataPtr&
Handle::WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz)
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    cl_int status = clEnqueueWriteBuffer(
            cmQ, ddata.get(), CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "OpenCL error writing to buffer: " + std::to_string(sz));
    }
    return ddata;
}

void Handle::ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz)
{
    MIOPEN_HANDLE_LOCK
    this->Finish();
    auto status = clEnqueueReadBuffer(
            cmQ, ddata.get(), CL_TRUE, 0, sz, data, 0, nullptr, nullptr);
    if(status != CL_SUCCESS)
    {
        MIOPEN_THROW_CL_STATUS(status, "OpenCL error reading from buffer: " + std::to_string(sz));
    }
}

cl_context Handle::clContext = NULL;
cl_device_id Handle::device = NULL;
cl_command_queue Handle::cmQ = NULL;


}

