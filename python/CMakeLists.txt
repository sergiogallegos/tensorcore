# Python bindings configuration
pybind11_add_module(tensorcore_core tensorcore_core.cpp)

# Link libraries
target_link_libraries(tensorcore_core PRIVATE
    tensorcore
    pybind11::module
    pybind11::lto
    pybind11::windows_extras
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES}
)

# Include directories
target_include_directories(tensorcore_core PRIVATE include)

# Compiler definitions
target_compile_definitions(tensorcore_core PRIVATE
    VERSION_INFO=${pybind11_VERSION}
)

# Set module properties
set_target_properties(tensorcore_core PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)

# Installation
install(TARGETS tensorcore_core
    DESTINATION .
    COMPONENT python
)
