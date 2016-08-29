using Knet

# iteration for one sample forw/back
function iter(f, sample; loss=softloss, gclip=0.0, tst=false, o...)
    reset!(f)
    ystack = Any[]
    sumloss = 0.0
    _, v, sp = sample
    v = reshape(v, length(v), 1)

    # language input
    d = full(sp)
    N = size(d,2)-1

    # visual input
    (tst?forw:sforw)(f, v; decoding=false)

    for j=1:N
        ygold = d[:,j+1:j+1]
        x1 = d[:,j:j]
        ypred = (tst?forw:sforw)(f, x1; decoding=true)
        l = loss(ypred, ygold)
        sumloss += l
        push!(ystack, ygold)
    end


    if tst
        return sumloss/N
    end

    # backprop
    while !isempty(ystack)
        ygold = pop!(ystack)
        sback(f, ygold, loss)
    end

    sback(f, nothing, loss)
    update!(f; gclip=gclip)
    reset!(f; keepstate=true)
end

# one epoch training
function train(f, data, voc; o...)
    for s in data
        iter(f, s; o...)
    end
end

# mean loss
function test(f, data, voc; o...)
    sumloss, numloss = 0.0, 0
    for s in data
        l = iter(f, s; tst=true, o...)
        sumloss += l
        numloss += 1
    end
    return sumloss/numloss
end
