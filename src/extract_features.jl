using Knet, ArgParse, JLD, MAT

LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]


function main(args)
    s = ArgParseSettings()
    s.description = "Extract CNN features of images (now only just for VGG-16)"

    @add_arg_table s begin
        ("--input"; help="image data file in JLD format")
        ("--model"; help="CNN model file")
        ("--output"; help="extracted features output file")
        ("--lastlayer"; default="fc7"; help="last layer for feature extraction")
        ("--batchsize"; default=64; help="batch size for extraction")
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # load data
    data = load(o[:input])
    bs, ll = o[:batchsize], o[:lastlayer]
    cnn = matread(o[:model])
    model = layers(cnn["layers"], ll)

    # bulk feature extraction
    for split in keys(data)
        data[split]["features"] = split_feature_extraction(data[split]["images"], bs, model)
        delete!(data[split], "images")
    end

    # save features
    save(o[:output], "data", data)
end


function split_feature_extraction(data, bs, model)
    ns = length(data) # number of samples
    nb = Int(ceil(ns/bs)) # number of batches
    rvecs2array(A) = map(i -> A[i,:], 1:size(A,1))
    nth_batch(data, n) = n != ns ? data[(n-1)*bs+1:n*bs] : data[(n-1)*bs+1:end]
    extract_features = get_feature_extractor(model...)
    features = []
    
    for i = 1:nb
        append!(features, extract_features(nth_batch(data,i)))
    end

    return features
end


function get_feature_extractor(ws, os, bs)
    function extract_features(xs)
        xs = KnetArray(xs)
        i, j = 1, 1
        nw, no = length(ws), length(os)
        while true
            if bs[i]
                xs = forw(xs, os[i],w s[j])
                j += 1
            else
                xs = forw(xs, os[i])
            end
        
            i += 1
            (i > no || j > nw) && break
        end
        return xs
    end
end


function layers(ls, ll) # layers and last layers
    ws, os, bs = [], [], [] # weights, ops, op have weight? arrays
    for l in ls
        lt(x) = startswith(l["name"],x)
        op = filter(x -> lt(x), LAYER_TYPES)[1]
        push!(os, op)
        push!(bs, haskey(l, "weights") && length(l["weights"]) != 0)
        bs[end] || continue
        
        w = l["weights"]
        if op == "conv"
            w[2] = reshape(w[2], (1,1,length(w[2]),1))
        elseif == "fc"
            w[1] = transpose(mat(w[1]))
        end
        push!(ws, w)

        lt(ll) && break
    end
    return map(KnetArray, ws), os, bs
end


convx(x,w) = conv4(w,x;padding=1,mode=1)
relux = relu
poolx = pool
fcx(x,w) = w[1]*x+w[2]
tofunc(op) = eval(parse(string(op, "x")))
forw(x,op) = tofunc(op)(x)
forw(x,op,w) = tofunc(op)(x,w)


!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
