module DeepDream

using Flux, Metalhead, Images, FileIO
try
    using CuArrays
catch
    warn("No GPU Support Available for your machine. Computation time shall be severely affected.")
end

export load_image, generate_image, save_image, load_model, deepdream

include("utils.jl")
include("dream.jl")
include("autodetect.jl")

end # module
