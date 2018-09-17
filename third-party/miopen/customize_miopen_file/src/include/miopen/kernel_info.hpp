/*
 * kernel_info.hpp
 *
 *  Created on: Sep 10, 2018
 *      Author: junpeng
 */

#ifndef SABER_FUNCS_IMPL_AMD_KERNEL_INFO_HPP_
#define SABER_FUNCS_IMPL_AMD_KERNEL_INFO_HPP_

#include <string>
#include <vector>

namespace miopen{
namespace solver{

struct KernelInfo
{
    std::string comp_options;
    std::vector<size_t> l_wk;
    std::vector<size_t> g_wk;
    std::string kernel_file;
    std::string kernel_name;
    friend std::ostream& operator<<(std::ostream& os, const KernelInfo& k);
};

}
}



#endif /* SABER_FUNCS_IMPL_AMD_KERNEL_INFO_HPP_ */
