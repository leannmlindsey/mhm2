set(CGRAPH_TARGET_OBJECTS)

foreach(tgt cgraph build_ctg_graph ctg_graph walk_ctg_graph spanner splinter)
  add_library(${tgt} OBJECT ${tgt}.cpp)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
    target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  endif()
  list(APPEND CGRAPH_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
endforeach()

add_library(CGRAPH_LIBRARY ${CGRAPH_TARGET_OBJECTS})
