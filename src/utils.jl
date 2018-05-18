# -------------------- Loading and Saving Image Utilities --------------------

function load_image(path, on_cpu=true)
    img = load(path)
    image_to_arr(img, on_cpu)
end

function generate_image(x)
    x = reshape(x, size(x,1), size(x,2), size(x,3))
    x = clamp.(permutedims(x, [3,2,1]), 0, 255) / 255
    colorview(RGB, x)
end

function save_image(path, x::Array)
    save(path, generate_image(x))
end

function image_to_arr(img, on_cpu = true; preprocess = true)
    local x = img
    if(on_cpu)
        x = float64.(channelview(img))
    else
        x = float.(channelview(img))
    end
    x = permutedims(x, [3,2,1])
    if(preprocess)
        local mean = [0.485, 0.456, 0.406]
        local std = [0.229, 0.224, 0.225]
        for i in 1:3
            x[i,:,:] = (x[i,:,:] - mean[i])/std[i]
        end
    end
    x = reshape(x, size(x,1), size(x,2), size(x,3), 1)
    x = (x * 255) |> gpu
end

# -------------------- Utilities to modify Images --------------------

function zoom_image(x, scale_x, scale_y)
    img = generate_image(x)
    inv_scalex = 1 / scale_x
    inv_scaley = 1 / scale_y
    local s = size(img)
    local r1 = Int(ceil(s[1] * (1 - inv_scalex) / 2)) : Int(ceil(s[1] * (1 + inv_scalex) / 2))
    local r2 = Int(ceil(s[2] * (1 - inv_scaley) / 2)) : Int(ceil(s[2] * (1 + inv_scaley) / 2))
    img = imresize(img[r1, r2], s)
    image_to_arr(img)
end

# -------------------- Loading the Neural Network Model --------------------

function load_model(layer, m = VGG19)
    model = m()
    global model = Chain(model.layers[1:layer]...) |> gpu
end
