# -------------------- Loading and Saving Image Utilities --------------------

im_mean = reshape([0.485, 0.456, 0.406], 1, 1, 3) |> gpu
im_std = reshape([0.229, 0.224, 0.225], 1, 1, 3) |> gpu

"""
    load_image(path, resize = false; size_save = true)

Load the original image as an array.

Arguments:
1. `path`: Path to the original image.
2. `resize`: Defaults to `false`. Setting it to `true` will reshape
             the original image to 224×224. This is recommended when
             using on `cpu`. However, when the image is finally saved
             the original dimensions are restored.
3. `size_save`: Set it to `false` when the size is not to be stored
                globally.
"""

function load_image(path, resize = false; size_save = true)
    img = load(path)
    if(size_save)
        global original_size = size(img)
    end
    if(resize)
        image_to_arr(imresize(img, (224, 224)))
    else
        image_to_arr(img)
    end
end

"""
    load_guide_image(path)

Utility to load the guide image. This essentially falls back to the
`load_image` function. The guide image is always resized to 224×224.
Simce most of the models used are trained on `Imagenet` the features are
best captured when the dimensions are 224×224.

Arguments:
1. `path`: Path to the guiding image. Ideally the guiding image should have
           a dimension close to 224×224 to ensure not a lot of features are
           lost while downscaling.
"""

load_guide_image(path) = load_image(path, true, size_save = false)

"""
    generate_image(x, resize_original = false)

Get back the original image from its array. It is assumed that the image was
preprocessed and is an RGB image.

Arguments:
1. `x`: Array from which the image must be generated.
2. `resize_original`: Set it to `true` if the original dimensions of the image
                      were changed
"""

function generate_image(x, resize_original = false)
    x = reshape(x, size(x)[1:3]...)
    x = x .* im_std .+ im_mean
    x = clamp.(permutedims(x, [3,2,1]), 0, 1) |> cpu
    if resize_original
        imresize(colorview(RGB, x), original_size)
    else
        colorview(RGB, x)
    end
end

"""
    save_image(path, x)

Function to save the generated image. Internally it simply makes a call to
the `generate_image` function.

Arguments:
1. `path`: The path where the image must be saved.
2. `x`: The array of the image to be saved.
"""

save_image(path, x) = save(path, generate_image(x, true))

"""
    image_to_arr(img; preprocess = true)

Converts an image to an array. Also applies preprocessing using the Imagenet
mean and standard deviation. The eltype of the array is always of type Float32.

Arguments:
1. `img`: Image whose array is to be generated.
2. `preprocess`: If set to `true` the image is preprocessed wrt Imagenet.
"""

function image_to_arr(img; preprocess = true)
    local x = img
    x = Float32.(channelview(img))
    x = permutedims(x, [3,2,1]) |> gpu
    if(preprocess)
        x = (x .- im_mean)./im_std
    end
    x = reshape(x, size(x,1), size(x,2), size(x,3), 1)
end

# -------------------- Utilities to modify Images --------------------

"""
    zoom_image(x, scale_x, scale_y)

Function to centre zoom into the image.

Arguments:
1. `x`: The array of the image.
2. `scale_x`: Scale along the x axis.
3. `scale_y`: Scale along the y axis.
"""

function zoom_image(x, scale_x, scale_y)
    img = generate_image(x)
    inv_scalex = 1 / scale_x
    inv_scaley = 1 / scale_y
    local s = size(img)
    local r1 = clamp(Int(ceil(s[1] * (1 - inv_scalex) / 2)), 1, s[1]) : clamp(Int(ceil(s[1] * (1 + inv_scalex) / 2)), 1, s[1])
    local r2 = clamp(Int(ceil(s[2] * (1 - inv_scaley) / 2)), 1, s[2]) : clamp(Int(ceil(s[2] * (1 + inv_scaley) / 2)), 1, s[2])
    img = imresize(img[r1, r2], s)
    image_to_arr(img)
end

# -------------------- Loading the Neural Network Model --------------------

"""
    load_model(layer, m = VGG19)

Function that globally loads the model.

Arguments:
1. `layer`: The layer upto which the model is to be imported.
2. `m`: The model to be loaded. All `Metalhead.jl` models are supported by
        default. To use a custom model make sure that the layers of the model
        are indexable. For more information about this look at Metalhead.jl.
"""

function load_model(layer, m = VGG19)
    model = m()
    global model = Chain(model.layers[1:layer]...) |> gpu
end

# ---------------- Utilities for Calculations on Matrices -------------------

"""
    argmax(A, dims)

Find the maximum index along a particular dimension

Arguments:
1. `A`: The matrix
2. `dims`: The dimension along which the maxima needs to be found.
"""

function argmax(A, dims)
   z = findmax(A, dims)[2] .% size(A, dims)
   z[z.==0] .= size(A,dims)
   z
end

#-----------------Utilities to apply Deepdream on a Batch--------------------

"""
    dream_batch(depth, iterations, η, octave_scale, num_octaves, guide = 1.0; path = "../images/")

Utility to generate dreams on a batch of images. This is a recommended method to be used
when applying deepdream on a bunch of images without varying the hyperparameters.

For individual parameter description look into the documentation of `deepdream` function.
"""

function dream_batch(depth, iterations, η, octave_scale, num_octaves, guide = 1.0; path = "../images/")
    images = [joinpath(path, i) for i in readdir(path)]
    load_model(depth)
    for i in images
        save_path = *((split(i, ".")[1:end-1])...) * "_dream.jpg"
        deepdream(load_image(i), iterations, η, octave_scale, num_octaves, save_path, guide)
        info("Image saved at $(save_path)")
    end
end

