using ArgParse, JLD, MAT

include("convnet.jl")


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


!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
