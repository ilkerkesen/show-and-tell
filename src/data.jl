using MAT, JSON

# for karpathy data
function build_data(vgg_filename, json_filename)
    vgg_fs = matread(vgg_filename)["feats"]
    json_data = JSON.parsefile(json_filename)
    images = json_data["images"]
    data, words = Dict(), Set()

    # data split, also build words set
    for i in 1:length(images)
        !haskey(data, images[i]["split"]) && push!(data, images[i]["split"] => [])
        push!(data[images[i]["split"]], (vgg_fs[:,i], images[i]))
    end

    # prepare words
    words = mapreduce(i -> mapreduce(s -> s["tokens"], vcat, i[2]["sentences"]),
                      vcat,
                      data["train"])

    # build vocabulary
    voc = Vocabulary(words)

    # build sentences
    helper(a) = mapreduce(e -> map(s -> (e[2]["filename"], e[1], sen2vec(voc, s["tokens"])), e[2]["sentences"]), vcat, a)
    trn, val, tst = map(i -> helper(data[i]), ["train", "val", "test"])

    return (data, voc, trn, val, tst)
end
