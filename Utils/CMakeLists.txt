
add_library(Utils)

target_sources(Utils 
  PRIVATE
	Exception.h
	IDFactory.cpp
	IDFactory.h
	Preprocessor.h
	Timing.cpp
	Timing.h	
	vec_math.h
    MortonCode.h)

target_include_directories(Utils
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
)


target_link_libraries(Utils 
	PUBLIC 
	CUDA::cudart
	CUDA::cuda_driver 
	#CUDA::cudart_static
)

set_property(TARGET Utils PROPERTY CUDA_ARCHITECTURES OFF)

if (WIN32)
  target_compile_definitions(Utils PUBLIC -DNOMINMAX)
endif()