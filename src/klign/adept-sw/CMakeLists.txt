if(NOT ENABLE_CUDA)
  message(FATAL_ERROR "Trying to build ADEPT-SW but CUDA is not enabled")
endif()

add_library(ADEPT_SW_LIBRARY_obj OBJECT driver.cpp kernel.cpp)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13)
  target_link_libraries(ADEPT_SW_LIBRARY_obj INTERFACE)
endif()

set_source_files_properties(driver.cpp kernel.cpp
                            PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)
option(ADEPT_SW_SHARED "Adept-SW shared library" OFF)
option(ADEPT_SW_STATIC "Adept-SW static library" ON)

if(ADEPT_SW_SHARED)
  add_library(ADEPT_SW_LIBRARY_shared SHARED
              $<TARGET_OBJECTS:ADEPT_SW_LIBRARY_obj>)
  set_property(TARGET ADEPT_SW_LIBRARY_shared
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(ADEPT_SW_LIBRARY_shared INTERFACE)
  install(TARGETS ADEPT_SW_LIBRARY_shared LIBRARY DESTINATION lib)
endif()
if(ADEPT_SW_STATIC)
  add_library(ADEPT_SW_LIBRARY_static STATIC
              $<TARGET_OBJECTS:ADEPT_SW_LIBRARY_obj>)
  target_link_libraries(ADEPT_SW_LIBRARY_static INTERFACE)
  set_property(TARGET ADEPT_SW_LIBRARY_static
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  install(TARGETS ADEPT_SW_LIBRARY_static ARCHIVE DESTINATION lib)
endif()
