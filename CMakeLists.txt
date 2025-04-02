cmake_minimum_required(VERSION 3.15)
project(vkbtest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

set(SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
set(OUTPUT_DIR ${CMAKE_SOURCE_DIR}/out)
set(ASSETS_DIR ${CMAKE_SOURCE_DIR}/resources)

set(PYTHON_EXECUTABLE py)

include_directories(
    C:/VulkanSDK/1.4.304.1/Include
    C:/Users/macie/Git/vcpkg/installed/x64-windows-static/include
)

link_directories(
    C:/Users/macie/Git/vcpkg/installed/x64-windows-static/lib
)

# source files
set(SOURCE_FILES 
    ${SOURCE_DIR}/main.cpp 
    ${SOURCE_DIR}/third_party/VkBootstrap.cpp
)

# output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# Define shaders and their compilation settings (file, entry point, target profile)
set(SHADER_LIST
    "${SOURCE_DIR}/shaders/vertex.glsl|vertex|${OUTPUT_DIR}/vertex.spv|vertex.h"
    "${SOURCE_DIR}/shaders/fragment.glsl|fragment|${OUTPUT_DIR}/fragment.spv|fragment.h"
)

set(ASSETS_LIST
    "${ASSETS_DIR}/bricks.png|bricks.h"
)

# generate shader compile command
function(compile_shader SHADER_FILE TARGET OUTPUT_FILE HEADER_FILE)
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND glslc -fshader-stage=${TARGET} ${SHADER_FILE} -o ${OUTPUT_FILE}
        DEPENDS ${SHADER_FILE}
        COMMENT "Compiling shader: ${SHADER_FILE} (Target: ${TARGET}) -> ${OUTPUT_FILE}"
        COMMAND py ${CMAKE_SOURCE_DIR}/embed_file.py ${OUTPUT_FILE} ${SOURCE_DIR}/resources/${HEADER_FILE}
        COMMENT "Create binary include: ${OUTPUT_FILE} -> ${SOURCE_DIR}/resources/${HEADER_FILE}"
        VERBATIM
    )
    set(SHADER_OUTPUTS ${SHADER_OUTPUTS} ${OUTPUT_FILE} PARENT_SCOPE)
endfunction()

function(compile_asset ASSET_FILE HEADER_FILE)
    SET(OUTPUT_FILE ${SOURCE_DIR}/resources/${HEADER_FILE})
    add_custom_command(
        OUTPUT ${OUTPUT_FILE}
        COMMAND py ${CMAKE_SOURCE_DIR}/embed_file.py ${ASSET_FILE} ${OUTPUT_FILE}
        COMMENT "Create binary asset: ${ASSET_FILE} -> ${OUTPUT_FILE}"
        VERBATIM
    )
    set(ASSET_OUTPUTS ${ASSET_OUTPUTS} ${OUTPUT_FILE} PARENT_SCOPE)
endfunction()

# compile all shaders from SHADER_LIST
set(SHADER_OUTPUTS "")
foreach(SHADER_DATA ${SHADER_LIST})
    string(REPLACE "|" ";" SHADER_PROPS ${SHADER_DATA})
    list(GET SHADER_PROPS 0 SHADER_FILE)
    list(GET SHADER_PROPS 1 TARGET)
    list(GET SHADER_PROPS 2 OUTPUT_FILE)
    list(GET SHADER_PROPS 3 HEADER_FILE)
    compile_shader(${SHADER_FILE} ${TARGET} ${OUTPUT_FILE} ${HEADER_FILE})
endforeach()

set(ASSET_OUTPUTS "")
foreach(ASSET_DATA ${ASSETS_LIST})
    string(REPLACE "|" ";" ASSET_PROPS ${ASSET_DATA})
    list(GET ASSET_PROPS 0 ASSET_FILE)
    list(GET ASSET_PROPS 1 ASSET_HEADER)
    compile_asset(${ASSET_FILE} ${ASSET_HEADER})
endforeach()

# add executable
add_executable(vkbtest ${SOURCE_FILES} ${SHADER_OUTPUTS} ${ASSET_OUTPUTS})

# use statically linked runtime on windows
set_property(TARGET vkbtest PROPERTY MSVC_RUNTIME_LIBRARY MultiThreaded) # /MT

target_link_libraries(vkbtest glfw3)

# linking
# if(WIN32)
#     target_link_options(vkbtest PRIVATE "/SUBSYSTEM:WINDOWS")
#     target_link_libraries(vkbtest user32.lib kernel32.lib d3d11.lib)
# endif()
