include(ExternalProject)

set(MIOPEN_PROJECT       extern_miopen)
set(MIOPEN_PREFIX_DIR    ${ANAKIN_TEMP_THIRD_PARTY_PATH}/miopen)
set(MIOPEN_INSTALL_ROOT  ${ANAKIN_THIRD_PARTY_PATH}/miopen)
set(MIOPEN_SOURCE_DIR    ${MIOPEN_PREFIX_DIR}/src/${MIOPEN_PROJECT})
set(MIOPEN_BINARY_DIR    ${MIOPEN_PREFIX_DIR}/src/${MIOPEN_PROJECT}-build)
set(MIOPEN_LIB           ${MIOPEN_INSTALL_ROOT}/lib/libMIOpen.so  CACHE FILEPATH "miopen library." FORCE)

if(NOT Boost_FOUND)
    set(BOOST_ROOT       ${BOOST_INSTALL_ROOT} CACHE FILEPATH "boost library/" FORCE)
endif()

message(STATUS "Scanning external modules ${Green}MIOPEN${ColourReset} ...")

ExternalProject_Add(
    ${MIOPEN_PROJECT}
    DEPENDS               boost
    GIT_REPOSITORY        https://github.com/ROCmSoftwarePlatform/MIOpen.git
    GIT_TAG               1.4.2
    #URL                   https://github.com/ROCmSoftwarePlatform/MIOpen/archive/1.4.2.zip
    #URL_HSA               SHA256=8c6dd7b3030c2ca3aea68f070daf5a55ce6a0e58642fa466bf2a09eaf51613aa
    PREFIX                ${MIOPEN_PREFIX_DIR}
    CMAKE_ARGS            -DMIOPEN_BACKEND=OpenCL -DCMAKE_INSTALL_PREFIX=${MIOPEN_INSTALL_ROOT} -DCMAKE_INSTALL_LIBDIR=lib -DBOOST_ROOT=${BOOST_ROOT}
    #LOG_DOWNLOAD          1
    LOG_BUILD             1
    
)

ExternalProject_Add_Step(
    ${MIOPEN_PROJECT} ${MIOPEN_PROJECT}_customize
    DEPENDEES         download
    DEPENDERS         build
    COMMAND           cp ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file/CMakeLists.txt ${MIOPEN_SOURCE_DIR}
    ALWAYS            1
    EXCLUDE_FORM_MAIN 1
    LOG               1
)

ExternalProject_Add_Step(
    ${MIOPEN_PROJECT} ${MIOPEN_PROJECT}_customize_2
    DEPENDEES         download
    DEPENDERS         build
    COMMAND           cp ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file/src/CMakeLists.txt ${MIOPEN_SOURCE_DIR}/src
    ALWAYS            1
    EXCLUDE_FORM_MAIN 1
    LOG               1
)

ExternalProject_Add_Step(
    ${MIOPEN_PROJECT} ${MIOPEN_PROJECT}_customize_3
    DEPENDEES         download
    DEPENDERS         build
    COMMAND           find ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file/src/ -name *.cpp -exec cp {} ${MIOPEN_SOURCE_DIR}/src \\$<SEMICOLON>
    ALWAYS            1
    EXCLUDE_FORM_MAIN 1
    LOG               1
)

ExternalProject_Add_Step(
    ${MIOPEN_PROJECT} ${MIOPEN_PROJECT}_customize_4
    DEPENDEES         download
    DEPENDERS         build
    COMMAND           find ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file/src/include/miopen/ -name *.hpp -exec cp {} ${MIOPEN_SOURCE_DIR}/src/include/miopen \\$<SEMICOLON>
    ALWAYS            1
    EXCLUDE_FORM_MAIN 1
    LOG               1
)

ExternalProject_Add_Step(
    ${MIOPEN_PROJECT} ${MIOPEN_PROJECT}_customize_5
    DEPENDEES         download
    DEPENDERS         build
    COMMAND           find ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file/src/solver/ -name *.cpp -exec cp {} ${MIOPEN_SOURCE_DIR}/src/solver \\$<SEMICOLON>;
    ALWAYS            1
    EXCLUDE_FORM_MAIN 1
    LOG               1
)

ExternalProject_Add_Step(
    ${MIOPEN_PROJECT} ${MIOPEN_PROJECT}_customize_6
    DEPENDEES         download
    DEPENDERS         build
    COMMAND           find ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file/src/ocl/ -name *.cpp -exec cp {} ${MIOPEN_SOURCE_DIR}/src/ocl/ \\$<SEMICOLON>;
    ALWAYS            1
    EXCLUDE_FORM_MAIN 1
    LOG               1
)

ExternalProject_Add_Step(
    ${MIOPEN_PROJECT} ${MIOPEN_PROJECT}_customize_7
    DEPENDEES         download
    DEPENDERS         build
    COMMAND           find ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file/src/kernels/ -name *.cl -exec cp {} ${MIOPEN_SOURCE_DIR}/src/kernels/ \\$<SEMICOLON>;
    ALWAYS            1
    EXCLUDE_FORM_MAIN 1
    LOG               1
)

ExternalProject_Add_Step(
    ${MIOPEN_PROJECT} ${MIOPEN_PROJECT}_customize_8
    DEPENDEES         download
    DEPENDERS         build
    COMMAND           find ${ANAKIN_THIRD_PARTY_PATH}/miopen/customize_miopen_file/src/kernels/ -name *.so -exec cp {} ${MIOPEN_SOURCE_DIR}/src/kernels/ \\$<SEMICOLON>;
    ALWAYS            1
    EXCLUDE_FORM_MAIN 1
    LOG               1
)
include_directories(${MIOPEN_INSTALL_ROOT}/include)
add_library(miopen SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET miopen PROPERTY IMPORTED_LOCATION ${MIOPEN_LIB})
add_dependencies(miopen ${MIOPEN_PROJECT})

list(APPEND ANAKIN_SABER_DEPENDENCIES miopen)
list(APPEND ANAKIN_LINKER_LIBS ${MIOPEN_LIB})
