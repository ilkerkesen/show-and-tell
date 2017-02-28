using Knet
using ArgParse
using JLD
using MAT
using Images

include("lib/vocab.jl")
include("lib/convnet.jl")
include("lib/model.jl")
include("lib/imgproc.jl")

function main(args)
    s = ArgParseSettings()
    s.description = "Caption generation script for the model (raw images)"

    @add_arg_table s begin
        ("--image"; help="input image")
        ("--vocabfile"; help="vocab file")
        ("--modelfile"; help="trained model file")
        ("--cnnfile"; help="convnet file for non-finetuned model")
        ("--beamsize"; arg_type=Int; default=1)
        ("--maxlen"; arg_type=Int; default=30; help="max sentence length")
        ("--nogpu"; action=:store_true)
        ("--lastlayer"; default="relu7")
        ("--imsize"; arg_type=Int; nargs=2; default=[224,224];
         help="new image sizes")
        ("--rgbmean"; arg_type=Float32; nargs=3;
         default=map(Float32, [123.68, 116.779, 103.939]))
    end

    # parse args
    println("Datetime: ", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o); flush(STDOUT)

    newsize = tuple(o[:imsize]...)
    rgbmean = reshape(o[:rgbmean], (1,1,3))

    # load model
    atype = o[:nogpu] ? Array{Float32} : KnetArray{Float32}
    w = load(o[:modelfile], "w")
    vocab = load(o[:vocabfile], "vocab")
    w = map(i->convert(atype, i), w)
    lossval = load(o[:modelfile], "lossval")
    s = initstate(atype, size(w[3], 1), 1)

    # load cnn
    wcnn = load(o[:modelfile], "wcnn")
    if wcnn != nothing
        wcnn = map(i->convert(atype, i), wcnn)
    elseif o[:cnnfile] != nothing
        vggmat = matread(abspath(o[:cnnfile]))
        wcnn = get_vgg_weights(vggmat; last_layer=o[:lastlayer])
    end
    wcnn == nothing && error("CNN is a MUST")
    @printf("Data and model loaded [%s]\n", now()); flush(STDOUT)

    # generate caption
    ti = now()
    @printf("Generation started (loss=%g,date=%s)\n", lossval, ti)
    flush(STDOUT)
    img = load(abspath(o[:image]))
    img = process_image(img, newsize, rgbmean)
    generated = generate(
        w, wcnn, copy(s), img, vocab; maxlen=o[:maxlen], beamsize=o[:beamsize])
    report_generation(o[:image], generated, o[:beamsize])
    tf = now()
    @printf("\nTime elapsed: %s [%s]\n", tf-ti, tf)
end

function report_generation(filename, generated, beamsize)
    @printf("\nFilename: %s\n", filename)
    @printf("Generated: %s\n", generated)
    @printf("Beamsize: %d\n", beamsize)
    flush(STDOUT)
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
