# DeepDream

**NOTE : If there is no GPU support available please checkout the cpu
branch of this repo. The cpu code is not thoroughly tested.**

## USAGE INSTRUCTIONS

1. Clone the Repository
`git clone https://github.com/avik-pal/DeepDream.jl.git DeepDream`

2. Move into the directory and run julia

3. Include the file in julia
```julia
include("src/DeepDream.jl")
```

4. Inorder to generate dreams without using octaves run the following
   command with your own parameters
```julia
julia> img = load_image("./examples/sky.jpg")
julia> load_model(5)
julia> DeepDream.make_step(img, 10, 1.5, true, "./examples/sky_dream_new.jpg")
```
Make sure to pass all the arguments to the make_step function call to
avoid errors. Refer to the [function definition](https://github.com/avik-pal/DeepDream.jl/blob/11ef038ec6333114e521c6d6b422a4831c6bb0c8/src/dream.jl#L5) to understand what each parameter means.

5. To make use of octaves run the following commands
```julia
julia> img = load_image("./examples/sky.jpg")
julia> load_model(5)
julia> deepdream(img, 10, 1.5, 1.4, 4, "./examples/sky_dream_new.jpg")
```
Also be sure to checkout the [function definition](https://github.com/avik-pal/DeepDream.jl/blob/11ef038ec6333114e521c6d6b422a4831c6bb0c8/src/dream.jl#L27)

6. Incase you want to use any other model than the VGG19 model make sure
   to pass a function to `load_model()` which returns the model you want
   to use

## SOME EXAMPLES
|Original Image|Generated Image|
|:---:|:---:|
|![Sky](./examples/sky.jpg)|![Deepdream on Sky](./examples/sky_dream.jpg)|
|![Game](./examples/game.jpg)|![Deepdream on Game](./examples/game_dream.jpg)|
|![Rio](./examples/rio.jpg)|![Deepdream on Rio](./examples/rio_dream.jpg)|


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
