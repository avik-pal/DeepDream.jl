# -------------------- Loading and Saving Image Utilities --------------------

im_mean = reshape([0.485, 0.456, 0.406], 1, 1, 3) |> gpu
im_std = reshape([0.229, 0.224, 0.225], 1, 1, 3) |> gpu

function load_image(path, resize = false)
    img = load(path)
    global original_size = size(img)
    if(resize)
        image_to_arr(imresize(img, (224, 224)))
    else
        image_to_arr(img)
    endS
end

load_guide_image(path) = load_image(path, true)

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

function save_image(path, x)
    save(path, generate_image(x, true))
end

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

function load_model(layer, m = VGG19)
    model = m()
    global model = Chain(model.layers[1:layer]...) |> gpu
end

# ---------------- Utilities for Calculations on Matrices -------------------

function argmax(A, dims)
   z = findmax(A, dims)[2] .% size(A, dims) 
   z[z.==0] .= size(A,dims)
   z
end

#-----------------Utilities to apply Deepdream on a Batch--------------------

function dream_batch(depth, iterations, η, octave_scale, num_octaves, guide = 1.0; guided = false)
    if(guide==1.0 && guided)
        error("Guide Image not entered")
    end
    path = "../images/"
    images = [joinpath(path, i) for i in readdir(path)]
    load_model(depth)
    for i in images
        save_path = *((split(i, ".")[1:end-1])...) * "_dream.jpg"
        deepdream(load_image(i), iterations, η, octave_scale, num_octaves, save_path, guide, guided)
        info("Image saved at $(save_path)")
    end
end

