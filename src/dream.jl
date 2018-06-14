# -------------------- Dreaming Utilities --------------------

"""
    make_step(img, iterations, η, solo_call = false; path = "")

The core function of the deepdream. This is responsible for the forward
pass of the image through the model and generate the dream. At its core it
maximizes the `L2 Norm` of the image.

Arguments:
1. `img`: An array of the image on which to dream.
2. `iteratons`: Number of iterations. Ideally should be close to 10
3. `η`: Learning Rate. Taking a very high η will lead to divergence.
        Something close to 0.03 is recommended.
4. `solo_call`: Set it to `true` if this function is called indepently.
5. `path`: Keyword argument which **MUST** be specified if `solo_call`
           is set to `true`. This is the path where the final image is
           saved.
"""

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

"""
    deepdream(base_img, iterations, η, octave_scale, num_octaves, path, guide = 1.0)

Iteratively calls the `make_step` or the `guided_step` function depending on the
arguments. First it generates the `octaves` which are essentially `centre-zoomed`
parts of the `base_img`.

Arguments:
1. `base_img`: An array of the image on which to dream.
2. `iteratons`: Number of iterations. Ideally should be close to 10.
3. `η`: Learning Rate. Taking a very high η will lead to divergence.
        Something close to 0.03 is recommended.
4. `octave_scale`: Amount of zoom to be applied to the base image.
        Ideally taken to abe around 2.0.
5. `num_octaves` : Total number of octaves to be generated.
6. `path`: This is the path where the final image is saved.
7. `guide`: Pass the guiding image array.
"""

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

"""
    guided_step(img, guide, iterations, η, solo_call; path = "")

Core function to generate guided dreams. The `deepdream` function falls back
to this function to generate the dream using the guide. Internally it calculates
the similarity between the img feature vector and the guide feature vector by
computing their dot product and then maximizes the activation corresponding to
those features.

Arguments:
1. `img`: An array of the image on which to dream.
2. `guide`: The guiding image array.
3. `iteratons`: Number of iterations. Ideally should be close to 10
4. `η`: Learning Rate. Taking a very high η will lead to divergence.
        Something close to 0.03 is recommended.
5. `solo_call`: Set it to `true` if this function is called indepently.
6. `path`: Keyword argument which **MUST** be specified if `solo_call`
           is set to `true`. This is the path where the final image is
           saved.
"""

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
