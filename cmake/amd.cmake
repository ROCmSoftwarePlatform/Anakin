# Copyright (c) 2018 Anakin Authors, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
macro(amd_set_opencl_path)
    if(NOT DEFINED OpenCL_INCLUDE_DIR)
        set(OpenCL_INCLUDE_DIR "/opt/rocm/opencl/include")
    endif()
    if(NOT DEFINED OpenCL_LIBRARY)
        set(OpenCL_LIBRARY "/opt/rocm/opencl/lib/x86_64/libOpenCL.so")
    endif()
    
    #FIND_PACKAGE(OpenCL REQUIRED)
    #if(OpenCL_FOUND)
    #    message(STATUS "Found OpenCL in ${OpenCL_INCLUDE_DIRS}")
    #    message(STATUS "Found OpenCL lib in ${OpenCL_LIBRARIES}")
    #    include_directories(${OpenCL_INCLUDE_DIRS})
    #    LINK_LIBRARIES(${OpenCL_LIBRARIES})
    #endif()
endmacro()

macro(amd_build_cl_file file_path dest_path)
    FILE(GLOB CL_FILES ${file_path}/*.cl)
    message(STATUS "found cl files: ${CL_FILES}")
    foreach(src_file ${CL_FILES})
        get_filename_component(src_file_name ${src_file} NAME)
        message(STATUS "copy ${src_file} to : ${dest_path}/${src_file_name}")
        configure_file( ${absdir}/${src_file} ${dest_path}/${src_file_name} COPYONLY)
    endforeach()
endmacro()


macro(amd_build_cl_binary_file file_path dest_path)
    FILE(GLOB CL_FILES ${file_path}/*.so)
    message(STATUS "found cl files: ${CL_FILES}")
    foreach(src_file ${CL_FILES})
        get_filename_component(src_file_name ${src_file} NAME)
        message(STATUS "copy ${src_file} to : ${dest_path}/${src_file_name}")
        configure_file( ${absdir}/${src_file} ${dest_path}/${src_file_name} COPYONLY)
    endforeach()
endmacro()

macro(_amd_find_boost)
    find_package(Boost 1.66.0 QUIET COMPONENTS system filesystem)
    if(Boost_FOUND)
        Message(STATUS "Found Boost(${Boost_VERSION}) ${Boost_INCLUDE_DIRS}")
        include_directories(${Boost_INCLUDE_DIRS})
        link_directories(${Boost_LIBRARY_DIRS})
        add_library(boost INTERFACE)
        target_link_libraries(boost INTERFACE ${Boost_LIBRARIES})
        list(APPEND ANAKIN_SABER_DEPENDENCIES boost)
        list(APPEND ANAKIN_LINKER_LIBS boost)
    endif()
endmacro()

macro(amd_find_boost)
    _amd_find_boost()
    if(NOT Boost_FOUND)
        Message(STATUS "Boost not found in system")
        set(BOOST_ROOT ${ANAKIN_THIRD_PARTY_PATH}/boost)
        _amd_find_boost()
        if(NOT Boost_FOUND)
            Message(STATUS "Boost not found")
            include(cmake/external/boost.cmake)
        endif()
    endif()
endmacro()

macro(amd_set_miopengemm_path)
    if(NOT DEFINED MIOPENGEMM_INCLUDE_DIR)
        set(MIOPENGEMM_INCLUDE_DIR "/opt/rocm/miopengemm/include")
    endif()
    if(NOT DEFINED MIOPENGEMM_LIBRARY)
        set(MIOPENGEMM_LIBRARY "/opt/rocm/miopengemm/lib64/libmiopengemm.so")
    endif()

    #FIND_PACKAGE(miopengemm REQUIRED)
    #if(miopengemm_FOUND)
    #    message(STATUS "Found miopengemm in ${MIOPENGEMM_INCLUDE_DIR}")
    #    message(STATUS "Found miopengemm lib in ${MIOPENGEMM_LIBRARY}")
    #    include_directories(${MIOPENGEMM_INCLUDE_DIR})
    #    LINK_LIBRARIES(${MIOPENGEMM_LIBRARY})
    #endif()
endmacro()

