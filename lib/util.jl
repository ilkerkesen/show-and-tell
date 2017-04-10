# Knet array to Float32 array conversion
karr2arr(ws) = map(w -> convert(Array{Float32}, w), ws)

function extract_file_from_tar(tar, from, to)
    sc = count(x->x=='/', from[1:end-1])
    run(`tar -xf $tar --strip-components=$sc -C $to --directory $from`)
end

function extract_file_from_zip(zip, from, to)
    filecontent = readstring(`unzip -p $zip $from`)
    write(to, filecontent)
end

# FIXME: pretty elapsed time conversion
function pretty_time(t)
    t = round(t)
    h = Int(div(t, 3600))
    m = Int(div(mod(t, 3600), 60))
    s = Int(mod(mod(t, 3600), 60))

    retval = ""
    h != 0 && (retval = string(retval, h, " hours "))
    m != 0 && (retval = string(retval, m, " minutes "))
    s != 0 && (retval = string(retval, s, " seconds"))

    return retval
end

function convert_weight(atype, w::Dict)
    w = convert(Dict{Any,Any}, w)
    for k in keys(w)
        w[k] = convert_weight(atype, w[k])
    end
    return w
end

function convert_weight(atype, w::Array)
    map(i->convert_weight(atype,w[i]), [1:length(w)...])
end

function convert_weight{T<:Number}(atype, w::Union{KnetArray{T},Array{T}})
    return convert(atype, w)
end

function get_weights(o)
    if o[:loadfile] == nothing
        w = initweights(o)
    else
        w = load(o[:loadfile], "w")
        w = convert_weight(o[:atype], w)
    end
    return w
end

function get_wcnn(o)
    if o[:finetune] && o[:cnnfile]
        return get_vgg_weights(o[:cnnfile]; last_layer=o[:lastlayer])
    elseif o[:finetune] && !o[:cnnfile]
        return map(i -> convert(o[:atype], i), load(o[:loadfile], "wcnn"))
    else
        return nothing
    end
end

function get_opts(o,w)
    if o[:loadfile] == nothing || o[:newoptimizer]
        return init_opts(o,w)
    elseif o[:loadfile] != nothing
        opts = load(o[:loadfile], "opts")
        return copy_opts(opts,w,false)
    end
end

function copy_opts(opt::Knet.Sgd, w, save)
    Sgd(;lr=opt.lr,gclip=opt.gclip)
end

function copy_opts(opt::Knet.Adam, w, save)
    optcopy = Adam()

    # scalar elements
    optcopy.lr = opt.lr
    optcopy.beta1 = opt.beta1
    optcopy.beta2 = opt.beta2
    optcopy.t = opt.t
    optcopy.eps = opt.eps
    optcopy.gclip = opt.gclip

    # array elements
    if save
        optcopy.fstm  = Array(opt.fstm)
        optcopy.scndm = Array(opt.scndm)
    else
        optcopy.fstm = convert(typeof(w), opt.fstm)
        optcopy.scndm = convert(typeof(w), opt.scndm)
    end

    return optcopy
end

function copy_opts(opt::Knet.Adagrad, w, save)
    optcopy = Adagrad()
    optcopy.lr = opt.lr
    optcopy.eps = opt.eps
    optcopy.gclip = opt.gclip
    optcopy.G = opt.G
    return optcopy
end

function copy_opts(opts::Dict, w::Dict, save)
    optcopy = Dict()
    for k in keys(w)
        optcopy[k] = copy_opts(opts[k], w[k], save)
    end
    return optcopy
end

function copy_opts(opts::Array, w::Array, save)
    map(i -> copy_opts(opts[i], w[i], save), [1:length(w)...])
end

function init_opts(o, w::Dict)
    opts = Dict()
    for (k,v) in w
        opts[k] = init_opts(o,v)
    end
    return opts
end

function init_opts(o, w::Array)
    map(i->init_opts(o,w[i]), [1:length(w)...])
end

function init_opts{T<:Number}(o, w::Union{KnetArray{T},Array{T}})
    (eval(parse(o[:optim])))(;lr=o[:lr],gclip=o[:gclip])
end

function copy_weights(w::Dict)
    Dict(k => copy_weights(v) for (k,v) in w)
end

function copy_weights(w::Array)
    map(i->copy_weights(w[i]), [1:length(w)...])
end

function copy_weights{T<:Number}(w::Union{KnetArray{T}, Array{T}})
    Array(w)
end
