include(ExternalProject)

set(BOOST_PROJECT       extern_boost)
set(BOOST_PREFIX_DIR    ${ANAKIN_TEMP_THIRD_PARTY_PATH}/boost)
set(BOOST_INSTALL_ROOT  ${ANAKIN_THIRD_PARTY_PATH}/boost)

set(BOOST_BOOTSTRAP     ${BOOST_PREFIX_DIR}/src/${BOOST_PROJECT}/bootstrap.sh)
set(BOOST_BUILD         ${BOOST_PREFIX_DIR}/src/${BOOST_PROJECT}/b2)

set(Boost_SYSTEM_LIBRARY     ${BOOST_INSTALL_ROOT}/lib/libboost_system.so  CACHE FILEPATH "boost system library." FORCE)
set(Boost_FILESYSTEM_LIBRARY ${BOOST_INSTALL_ROOT}/lib/libboost_filesystem.so  CACHE FILEPATH "boost filesystem library." FORCE)
set(Boost_LIBRARY_DIRS       ${BOOST_INSTALL_ROOT}/lib CACHE FILEPATH "boost library dirs." FORCE)
set(Boost_INCLUDE_DIRS       ${BOOST_INSTALL_ROOT}/include CACHE FILEPATH "boost include dirs." FORCE)

message(STATUS "Scanning external modules ${Green}BOOST${ColourReset} ...")

ExternalProject_Add(
    ${BOOST_PROJECT}
    #GIT_REPOSITORY        https://github.com/boostorg/boost.git
    #GIT_TAG               boost-1.66.0
    URL                   http://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
    URL_HASH              SHA256=bd0df411efd9a585e5a2212275f8762079fed8842264954675a4fddc46cfcf60
    BUILD_IN_SOURCE       1
    PREFIX                ${BOOST_PREFIX_DIR}
    CMAKE_COMMAND         ""
    CMAKE_ARGS            ""
    CONFIGURE_COMMAND     ${BOOST_BOOTSTRAP} --prefix=${BOOST_INSTALL_ROOT}
    BUILD_COMMAND         ${BOOST_BUILD} -j4 --with-thread --with-system --with-filesystem install
    UPDATE_COMMAND        ""
    PATCH_COMMAND         ""
    INSTALL_COMMAND       ""
)

add_library(boost SHARED IMPORTED GLOBAL)
add_dependencies(boost ${BOOST_PROJECT})
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
SET_PROPERTY(TARGET boost PROPERTY IMPORTED_LOCATION ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})
list(APPEND ANAKIN_SABER_DEPENDENCIES boost)
list(APPEND ANAKIN_LINKER_LIBS ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})
