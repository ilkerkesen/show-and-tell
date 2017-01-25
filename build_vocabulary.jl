using ArgParse, JLD, JSON
include("lib/vocab.jl");
SPLITS = ["train", "restval", "val", "test"]


function main(args)
    s = ArgParseSettings()
    s.description = "Convert common image captioning datasets to JLD format."

    @add_arg_table s begin
        ("--captions"; required=true;
         help="captions archive file path (karpathy)")
        ("--savefile"; required=true; help="output file in JLD format")
        ("--dataset"; default="flickr8k";
         help="dataset (flickr8k|flickr30k|coco)")
        ("--minoccur"; arg_type=Int; default=5;
         help="vocabulary minimum occurence threshold")
        ("--extradata"; action=:store_true;
         help="for karpathy COCO restval split")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    zip = abspath(o[:captions])
    file = joinpath(split(splitdir(zip)[2], ".")[1], "dataset.json")
    entries = JSON.parse(readstring(`unzip -p $zip $file`))["images"]
    vocabulary_splits = ["train"]
    extradata = false
    if o[:dataset] == "coco" && o[:extradata]
        extradata = true
        push!(vocabulary_splits, "restval")
    end
    vocab = build_vocabulary(entries, vocabulary_splits, o[:minoccur])

    # save process data
    save(o[:savefile],
         "vocab", vocab,
         "extradata", extradata)
end

function build_vocabulary(entries, splits, minoccur)
    words = []
    for entry in entries
        in(entry["split"], splits) || continue
        entry_words = mapreduce(s -> s["tokens"], vcat, entry["sentences"])
        push!(words, entry_words...)
    end
    return Vocabulary(words, minoccur)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
