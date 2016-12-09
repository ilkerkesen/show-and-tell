LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]

function get_feature_extractor(ws, os, bs, as, ds)
    dropvals = get_dropouts(os, ds)
    function extract_features(xs)
        xs = KnetArray(map(Float32, xs)) .- as
        i, j = 1, 1
        nw, no = length(ws), length(os)
        while i <= no && j <= nw
            if bs[i]
                xs = forw(xs, os[i], ws[j])
                j += 1
            else
                xs = forw(xs, os[i])
            end

            if dropvals != nothing && dropvals[i] > 0
                xs = dropx(xs, dropvals[i])
            end
        
            i += 1
        end
        convert(Array{Float32}, xs)
    end
end


function weights(CNN; last_layer=nothing) # CNN model and last layer
    ls = CNN["layers"]
    as = CNN["meta"]["normalization"]["averageImage"][:,:,end:-1:1]
    as = KnetArray(convert(Array{Float32}, as));
    ws, os, bs = [], [], [] # weights, ops and "op have weight?" arrays
    for l in ls
        lt(x) = startswith(l["name"], x) # get layer type
        op = filter(x -> lt(x), LAYER_TYPES)[1]
        push!(os, op)
        push!(bs, haskey(l, "weights") && length(l["weights"]) != 0)
        
        if bs[end]
            w = l["weights"]
            if op == "conv"
                w[2] = reshape(w[2], (1,1,length(w[2]),1))
            elseif op == "fc"
                w[1] = transpose(mat(w[1]))
            end
            push!(ws, w)
        end

        last_layer != nothing && lt(last_layer) && break
    end
    return map(w -> map(KnetArray, w), ws), os, bs, as
end

# just only for fully connected layers
function get_dropouts(layers, values)
    values != nothing || return nothing
    nlayers = length(layers)
    j = 1
    retval = map(Float32, zeros(length(nlayers)))
    vals = isa(values, Array) ? reverse(values) : values
    for i = 1:nlayers-1
        if layers[i] == "fc" && layers[i+1] == "relu"
            retval[i+1] = isa(vals, Array) ? pop!(values) : vals
        end

        length(vals) >= 1 || break
    end
end


convx(x,w) = conv4(w[1], x; padding=1, mode=1) .+ w[2]
relux = relu
poolx = pool
fcx(x,w) = w[1]*mat(x) .+ w[2]
tofunc(op) = eval(parse(string(op, "x")))
forw(x,op) = tofunc(op)(x)
forw(x,op,w) = tofunc(op)(x,w)
dropx(x,d) = x .* (rand!(similar(getval(x))) .> d) * (1/(1-d))
