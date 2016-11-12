using Knet, ArgParse, JLD, MAT

LAYER_TYPES = ["conv", "relu", "pool", "fc", "prob"]


function main(args)
    s = ArgParseSettings()
    s.description = "Extract CNN features of images (now only just for VGG-16)"

    @add_arg_table s begin
        ("--input"; help="image data file in JLD format")
        ("--model"; help="CNN model file")
        ("--output"; help="extracted features output file")
        ("--lastlayer"; default="relu7"; help="last layer for feature extraction")
        ("--batchsize"; arg_type=Int; default=10; help="batch size for extraction")
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    # load data
    @printf("Data and model loading... "); flush(STDOUT)
    data = load(o[:input])
    bs, ll = o[:batchsize], o[:lastlayer]
    CNN = matread(o[:model])
    model = weights(CNN; last_layer=ll)
    newdata = Dict()
    @printf("Done.\n"); flush(STDOUT)

    # bulk feature extraction
    for split in keys(data)
        @printf("Feature extraction for %s split... ", split); flush(STDOUT)
        features = split_feature_extraction(data[split]["images"], bs, model)
        filenames = data[split]["filenames"]
        newdata[split] = Dict("filenames" => filenames, "features" => features)
        @printf("Done.\n");flush(STDOUT)
    end

    # save features
    @printf("Save extracted features to output file... "); flush(STDOUT)
    save(o[:output],
         "trn", newdata["trn"],
         "val", newdata["val"],
         "tst", newdata["tst"])
    @printf("Done.\n"); flush(STDOUT)
end


function split_feature_extraction(data, bs, model)
    ns = length(data) # number of samples
    nb = Int(ceil(ns/bs)) # number of batches
    rvecs2array(A) = map(i -> A[i,:], 1:size(A,1))
    nth_batch(n) = data[(n-1)*bs+1:min(n*bs,ns)]
    make_batch(A) = reduce((x...) -> cat(4, x...), A[1], A[2:end])
    make_nth_batch(n) = make_batch(nth_batch(n))
    extract_features = get_feature_extractor(model...)
    extract_nth_batch_features(n) = extract_features(make_batch(nth_batch(n)))
    return mapreduce(extract_nth_batch_features, hcat, 1:nb)
end


function get_feature_extractor(ws, os, bs, as)
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
        
            i += 1
        end
        return convert(Array{Float32}, xs)
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


convx(x,w) = conv4(w[1], x; padding=1, mode=1) .+ w[2]
relux = relu
poolx = pool
fcx(x,w) = w[1]*mat(x) .+ w[2]
tofunc(op) = eval(parse(string(op, "x")))
forw(x,op) = tofunc(op)(x)
forw(x,op,w) = tofunc(op)(x,w)


!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
