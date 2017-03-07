using ArgParse
using JSON

function main(args)
    s = ArgParseSettings()
    s.description = "Caption generation script for the model (raw images)"

    @add_arg_table s begin
        ("--input"; help="input file in JSON format")
        ("--outdir"; help="output dir")
    end

    # parse args
    println("Datetime: ", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)
    outdir = abspath(o[:outdir])
    inputfile = abspath(o[:input])
    elements = JSON.parsefile(inputfile)["captions"]
    rfs = Array(Any, 5)
    for k = 0:1:4
        rfs[k+1] = open(joinpath(outdir,"reference$k"),"w")
    end
    hf = open(joinpath(outdir,"output"),"w")

    for el in elements
        write(hf, string(el["hypothesis"][1:end-2],"\n"))
        for k = 1:5
            write(rfs[k], string(el["references"][k][1:end-2], "\n"))
        end
    end

    for k = 1:5
        close(rfs[k])
    end
    close(hf)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
