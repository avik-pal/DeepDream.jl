# -------------------- Dreaming Utilities --------------------

function make_step(img, iterations, η, solo_call = false; path = "")
    if(solo_call && path == "")
        error("Image Save Path must be specified for solo calls")
    end
    input = param(img)
    for i in 1:iterations
        out = model(input)
        Flux.back!(out, out.data)
        input.data .= input.data + η * input.grad / mean(abs.(input.grad))
        info("$i iterations complete")
    end
    if(solo_call)
        save_image(path, input.data)
    end
    return input.data
end

function deepdream(base_img, iterations, η, octave_scale, num_octaves, path, guide = 1.0)
    octaves = [copy(base_img)]
    for i in 1:(num_octaves-1)
        push!(octaves, zoom_image(octaves[end], octave_scale, octave_scale))
    end
    detail = zeros(octaves[end])
    out = base_img
    for (i, oct) in enumerate(octaves[length(octaves):-1:1])
        info("OCTAVE NUMBER = $i")
        w, h = size(oct)[1:2]
        if(i > 1)
            w1, h1 = size(detail)[1:2]
            detail = zoom_image(detail, w1 / w, h1 / h)
        end
        input_oct = (oct + detail) |> gpu
        if(guide != 1.0)
            out = guided_step(input_oct, guide, iterations, η)
        else
            out = make_step(input_oct, iterations, η)
        end
        detail = out - oct
    end
    save_image(path, out)
end

function guided_step(img, guide, iterations, η, solo_call = false; path = "")
    if(solo_call && path == "")
        error("Image Save Path must be specified for solo calls")
    end
    input = param(img)
    out = model(input)
    guide_features = model(guide)
    ch = size(out.data, 3)
    y = reshape(guide_features, :, ch)
    for i in 1:iterations
        out = model(input)
        x = reshape(out.data , :, ch)
        dot_prod = (x * y') |> cpu
        result = y[argmax(dot_prod, 2), :]
        result = reshape(result, size(out.data))
        Flux.back!(out, result)
        input.data .= input.data + η * input.grad / mean(abs.(input.grad))
        info("$i iterations complete")
    end
    if(solo_call)
        save_image(path, input.data)
    end
    return input.data
end
