# Monte Carlo Incompressible SPH

Particle-based fluid simulation (MCISPH) with a Monte Carlo pressure solver integrated into the Smoothed Particle Hydrodynamics (SPH) framework. This repository contains an OptiX implementation of MCISPH with GPU ray tracing for simulating incompressible fluids. The technical details of this implementation are described in our paper, "Monte Carlo Incompressible SPH".

## Source Code Overview

`3rd/`: external libraries (partio, zlib)  
`cmake/`: `.cmake` files used by `CMakeLists.txt` to find libraries and set up the build system  
`Configs/`: scene configuration files (.json), fluid particle files (.bgeo) and boundary geometry files (.obj)  
`MCISPH/`: core code (SPH framework, Monte Carlo pressure solver)  
`Utils/`: useful tools  

## Requirement

The code was tested with the following configurations:
* CUDA 12.4
* Optix 8.0
* CMake 3.22.1
* Ubuntu 22.04.4, GCC 11.4.0, RTX 4090
* Windows 11, Visual Studio 2022, RTX 4090

The repository includes the following external libraries:
* [partio](https://github.com/wdas/partio) and [zlib](https://github.com/madler/zlib): for particle IO and manipulation
* [json](https://github.com/nlohmann/json): for configuration parsing
* [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader): for loading mesh geometry

## Compile and Run
Compile the program by running the following command:
```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```
After compilation, run the program with a scene configuration file (.json), for example:
```
./MCISPH/MCISPH ../Configs/Street.json
```
The output files (.bgeo) can be viewed using `PartioView` in [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH). They can also be reconstructed into mesh geometry (.obj) with [splashsurf](https://github.com/InteractiveComputerGraphics/splashsurf) and then rendered in [Blender](https://github.com/blender/blender).

## Acknowledgement

* The neighbor search algorithm is implemented using a ray-tracing approach based on [RTNN](https://github.com/horizon-research/rtnn).
* The Monte Carlo pressure solver is partially adapted from [VelMCFluids](https://github.com/rsugimoto/VelMCFluids).
* The SPH framework is partially based on [SPlisHSPlasH](https://github.com/InteractiveComputerGraphics/SPlisHSPlasH).

## License

The code is released under the MIT license. See the `LICENSE` file for the details.
