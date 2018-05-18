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
        input.data .= input.data + η * grad / mean(abs(grad))
    end
    if(solo_call)
        save_image(path, input.data)
    else
        return input.data
    end
end