using Knet

# iteration for one sample forw/back
function iter(f, sample; loss=softloss, gclip=0.0, test=false)
    reset!(f)
    ystack = Any[nothing]
    sumloss = 0.0
    _, v, sp = sample

    # visual input
    sforw(f, v; decoding=false)

    # language input
    N = size(sp,2)-1
    for j=1:N
        ygold = sp[:,j+1]
        ypred = (test?forw:sforw)(f, sp[:,j])
        sumloss += loss(ypred, ygold)
        test && push!(ystack, sp[:,j+1])
    end

    test && return sumloss/N

    # backprop
    while !isempty(ystack)
        ygold = pop!(ystack)
        sback(f, ygold, loss)
    end

    update!(f; gclip=gclip)
    reset!(f; keepstate=true)

    return sumloss/N
end

# one epoch training
train(f, data, voc) = map!(s -> iter(f, s), data)

# mean loss
test(f, data, voc) = mean(map!(s -> iter(f, s; test=true), data))
