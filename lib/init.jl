# initialize hidden and cell arrays
function initstate(atype, hidden, batchsize)
    state = Array(Any, 2)
    state[1] = zeros(batchsize, hidden)
    state[2] = zeros(batchsize, hidden)
    return map(s->convert(atype,s), state)
end

# initialize all weights of decoder network
function initweights(o::Dict)
    w = Dict()
    srnn, w["rnn"] = rnninit(o[:embed], o[:hidden])
    w["wsoft"] = o[:winit]*randn(o[:vocabsize], o[:hidden])
    w["bsoft"] = zeros(o[:vocabsize], 1)
    w["vemb"] = o[:winit]*randn(o[:embed], o[:visual][1])
    w["wemb"] = o[:winit]*randn(o[:embed], o[:vocabsize])
    return convert_weight(o[:atype], w), srnn
end
