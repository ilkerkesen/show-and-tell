using Knet

# iteration for one sample forw/back
function iter(f, sample; loss=softloss, gclip=0.0, tst=false)
    # @bp
    reset!(f)
    ystack = Any[nothing]
    sumloss = 0.0
    _, v, sp = sample
    v = reshape(v, length(v), 1)

    # language input
    d = full(sp)
    N = size(d,2)-1

    # visual input
    (tst?forw:sforw)(f, v, d[:,1]; decoding=false)

    for j=1:N
        ygold = d[:,j+1:j+1]
        x1 = d[:,j:j]
        ypred = (tst?forw:sforw)(f, v, x1; decoding=true)
        l = loss(ypred, ygold)
        sumloss += l
        push!(ystack, ygold)
    end

    # backprop
    retvals = Any[]
    if !tst
        while !isempty(ystack)
            ygold = pop!(ystack)
            retval = sback(f, ygold, loss; getdx=true)
            push!(retvals, retval)
        end

        update!(f; gclip=gclip)
        reset!(f; keepstate=true)
    end

    return sumloss/N
end

# one epoch training
function train(f, data, voc)
    for s in data
        iter(f, s)
    end
end

# mean loss
function test(f, data, voc)
    sumloss, numloss = 0.0, 0
    for s in data
        l = iter(f, s; tst=true)
        sumloss += l
        numloss += 1
    end
    return sumloss/numloss
end
