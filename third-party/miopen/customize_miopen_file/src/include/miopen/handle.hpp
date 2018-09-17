/*
 * handle.hpp
 *
 *  Created on: Aug 31, 2018
 *      Author: junpeng
 *      fake miopen like handle object
 */

#ifndef SABER_FUNCS_IMPL_AMD_HANDLE_HPP_
#define SABER_FUNCS_IMPL_AMD_HANDLE_HPP_

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <CL/cl.h>

#include <miopen/kernel.hpp>
#if 0
/*! @enum miopenStatus_t
 * Error codes that are returned by all MIOpen API calls.
*/
typedef enum {
    miopenStatusSuccess        = 0, /*!< No errors */
    miopenStatusNotInitialized = 1, /*!< Data not initialized. */
    miopenStatusInvalidValue   = 2, /*!< Incorrect variable value. */
    miopenStatusBadParm        = 3, /*!< Incorrect parameter detected. */
    miopenStatusAllocFailed    = 4, /*!< Memory allocation error. */
    miopenStatusInternalError  = 5, /*!< MIOpen failure. */
    miopenStatusNotImplemented = 6, /*!< Use of unimplemented feature. */
    miopenStatusUnknownError   = 7, /*!< Unknown error occurred. */
} miopenStatus_t;
#endif

#include <miopen/allocator.hpp>

namespace miopen {

#define MIOPEN_HANDLE_LOCK

struct Handle
{


    //Handle(cl_context _clContext, cl_device_id _device);
    Handle();
    ~Handle();

    static int setClEnv(cl_context _clContext, cl_device_id _device);
    static int clearClEnv();

    cl_command_queue GetStream() const;

    void Finish() const;

    std::size_t GetLocalMemorySize();
    std::size_t GetMaxComputeUnits();

    std::string GetDeviceName();

    Allocator::ManageDataPtr Create(std::size_t sz);
    Allocator::ManageDataPtr&
    WriteTo(const void* data, Allocator::ManageDataPtr& ddata, std::size_t sz);
    void ReadTo(void* data, const Allocator::ManageDataPtr& ddata, std::size_t sz);

    KernelInvoke AddKernel(const std::string& algorithm,
                           const std::string& network_config,
                           const std::string& program_name,
                           const std::string& kernel_name,
                           const std::vector<size_t>& vld,
                           const std::vector<size_t>& vgd,
                           const std::string& params,
                           std::size_t cache_index = 0);

    KernelInvoke Run(Kernel k);

    void EnableProfiling(bool enable = true);

    void SetProfilingResult(cl_event& e);

    float GetKernelTime() const;

    template <class T>
    Allocator::ManageDataPtr Create(std::size_t sz)
    {
        return this->Create(sz * sizeof(T));
    }

    template <class Container>
    Allocator::ManageDataPtr Write(const Container& c)
    {
        using type = typename Container::value_type;
        auto buf   = this->Create<type>(c.size());
        return std::move(
            this->WriteTo(reinterpret_cast<const void*>(c.data()), buf, c.size() * sizeof(type)));
    }

    template <class T>
    std::vector<T> Read(const Allocator::ManageDataPtr& ddata, std::size_t sz)
    {
        std::vector<T> result(sz);
        this->ReadTo(result.data(), ddata, sz * sizeof(T));
        return result;
    }

private:
    static cl_context clContext;
    static cl_device_id device;
    static cl_command_queue cmQ;

    Allocator allocator{};
    std::size_t localMemSize;
    std::size_t maxComputeUnitSize;
    std::string deviceName;
    bool enable_profiling  = false;
    float profiling_result = 0.0;
};


}

#endif /* SABER_FUNCS_IMPL_AMD_HANDLE_HPP_ */
