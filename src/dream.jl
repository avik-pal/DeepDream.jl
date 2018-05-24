# -------------------- Dreaming Utilities --------------------

# TODO: Add the random jitter effect

function make_step(img, iterations, η, solo_call = false; path = "")
    if(solo_call && path == "")
        error("Image Save Path must be specified for solo calls")
    end
    input = param(img)
    for i in 1:iterations
        out = model(input)
        loss = mean(abs2.(out))
        @show loss
        Flux.back!(loss)
        info("$i iterations complete")
        local grad = input.grad
        input.data .= input.data + η * grad / mean(abs.(grad))
    end
    if(solo_call)
        save_image(path, input.data)
    end
    return input.data
end

# TODO: Add code for guided dream

function deepdream(base_img, iterations, η, octave_scale, num_octaves, path)
    octaves = [copy(base_img)]
    for i in 1:(num_octaves-1)
        push!(octaves, zoom_image(octaves[end], octave_scale, octave_scale))
    end
    detail = zeros(octaves[end])
    out = base_img
    for (i, oct) in enumerate(octaves[length(octaves):-1:1])
        w, h = size(oct)[1:2]
        if(i > 1)
            w1, h1 = size(detail)[1:2]
            detail = zoom_image(detail, w1 / w, h1 / h)
        end
        input_oct = (oct + detail) |> gpu
        out = make_step(input_oct, iterations, η)
        detail = out - oct
    end
    out = out |> cpu
    save_image(path, out)
end

