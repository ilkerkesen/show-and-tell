using ArgParse
using JSON

include("lib/eval.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Evaluation script for generations"

    @add_arg_table s begin
        ("--generations"; required=true; help="generations JSON file")
        ("--meta"; action=:store_true; help="print meta data")
    end

    # parse args
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)

    # read JSON file
    results = JSON.parsefile(o[:generations])

    # print META data
    if o[:meta]
        println("META")
        for (k,v) in results["meta"]
            println(k, " => ", v)
        end
        flush(STDOUT)
    end

    hyp, ref = [], []
    for caption in results["captions"]
        push!(hyp, caption["hypothesis"])
        push!(ref, caption["references"])
    end
    results = 0; gc()

    # evaluate
    ti = now()
    @printf("Evaluation started (date=%s)\n", ti)
    scores, bp, hlen, rlen = bleu(hyp, ref)
    @printf("BLEU = %.1f/%.1f/%.1f/%.1f ",
            map(i->i*100,scores)...)
    @printf("(BP=%g, ratio=%g, hyp_len=%d, ref_len=%d)\n",
            bp, hlen/rlen, hlen, rlen)
    tf = now()
    @printf("Time elapsed: %s [%s]\n", tf-ti, tf)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
