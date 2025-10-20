# CMakeList.txt : CMake project for MCSPH, include source and define
# project specific logic here.
#

# Add source to this project's executable.

include_directories(${PROJECT_SOURCE_DIR})

include(embed_ptx)
embed_ptx(
  OUTPUT_TARGET
    FluidSolver_ptx
  SOURCES
	${CMAKE_CURRENT_SOURCE_DIR}/FluidSolver.cu
)

add_executable (MCISPH 
	main.cpp 
	Geometry.cuh
	FluidSolver.h
	FluidModel.h 
	FluidModel.cu
	Common.cuh
	SPHKernel.cuh
	MCISPH.h
	MCISPH.cu
)

set(PARTIO_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/3rd/partio/src/lib/include)

target_include_directories(MCISPH 
  PRIVATE 
	${PARTIO_INCLUDE_DIR}
	${PROJECT_SOURCE_DIR}
	${PROJECT_SOURCE_DIR}/3rd
)

target_link_libraries(MCISPH 
  PRIVATE 
  	FluidSolver_ptx
	partio
	Utils
	CUDA::cudart
	CUDA::cuda_driver 
)

target_compile_definitions(MCISPH PRIVATE CONFIG_PATH="${PROJECT_SOURCE_DIR}/Configs/")
