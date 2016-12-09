using ArgParse, JLD, Images, JSON
include("vocab.jl");
SPLITS = ["train", "restval", "val", "test"]


function main(args)
    s = ArgParseSettings()
    s.description = "Convert common image captioning datasets to JLD format."

    @add_arg_table s begin
        ("--captions"; required=true; help="captions archive file path (karpathy)")
        ("--savefile"; required=true; help="output file in JLD format")
        ("--dataset"; default="flickr8k"; help="dataset (flickr8k|flickr30k|coco)")
        ("--minoccur"; arg_type=Int; default=5; help="vocabulary minimum occurence threshold")
        ("--extradata"; action=:store_true; help="for karpathy COCO restval split")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    zip = abspath(o[:captions])
    file = joinpath(split(splitdir(zip)[2], ".")[1], "dataset.json")
    entries = JSON.parse(readstring(`unzip -p $zip $file`))["images"]
    vocabulary_splits = ["train"]
    o[:dataset] == "coco" && o[:extradata] && push!(vocabulary_splits, "restval")
    vocabulary = build_vocabulary(entries, vocabulary_splits)


    # save process data
    save(o[:savefile],
         "train", data["train"],
         "val", data["val"],
         "restval", data["restval"],
         "test", data["test"],
         "vocab", vocab)
end


function build_vocabulary(entries, splits)
    words = []
    for entry in entries
        in(entry["split"], splits) || continue
        entry_words = mapreduce(s -> s["tokens"], vcat, entry["sentences"])
        push!(words, entry_words...)
    end
    return Vocabulary(words, o[:minoccur])
end


function process_language(entries, vocab)
    data = Dict()
    for splitname in SPLITS
        entries = filter(e -> e["split"] == splitname, entries)
        for entry in entries
            sentences = map(
                s -> (s["raw"], sen2vec(vocab, s["tokens"])), entry["sentences"])
            push!(splitdata, (entry["filename"], sentences...))
        end

        data[splitname] = splitdata
    end
end


!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
