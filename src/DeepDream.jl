module DeepDream

using Flux, Metalhead, Images, FileIO
using CuArrays

export load_image, generate_image, save_image, load_model, deepdream, dream

include("utils.jl")
include("dream.jl")
include("autodetect.jl")

end # module
