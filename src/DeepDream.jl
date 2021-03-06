__precompile__()

module DeepDream

using Flux, Metalhead, Images, FileIO
using CuArrays

export load_image, generate_image, save_image,
	   load_model, deepdream, dream_batch,
	   load_guide_image, writevideo, recurdream

include("utils.jl")
include("dream.jl")

end # module
