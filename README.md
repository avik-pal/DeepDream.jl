# DeepDream

**NOTE : If there is no GPU support available please checkout the cpu
branch of this repo**

## USAGE INSTRUCTIONS



## SOME EXAMPLES
|Original Image|Generated Image|
|:---:|:---:|
|![Sky](./examples/sky.jpg)|![Deepdream on Sky](./examples/sky_dream.jpg)|
|![Game](./examples/game.jpg)|![Deepdream on Game](./examples/game_dream.jpg)|

## DEPENDENCIES

1. Flux.jl
2. Metalhead.jl
3. Images.jl
4. CuArrays.jl

## IMPLEMENTED

1. Utilities to load, save and generate images
2. Perform operations on Image
    * Zoom
3. Utilities to load models
4. Deep Dream Generator (non-guided)
5. Generate deep dreams using Octaves

## TODO

* Implement Guided Dreams
* Add automatic detection of images and perform deepdream on them
* Provide standard functions with predefined set of parameters

## CURRENT BOTTLENECKS

* The image zoom is performed in CPU as the present implementation is
    too slow for GPUs. So it can be quite slow
