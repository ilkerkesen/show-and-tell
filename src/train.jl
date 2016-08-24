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

    # backprop
    if !test
        while !isempty(ystack)
            ygold = pop!(ystack)
            sback(f, ygold, loss)
        end

        update!(f; gclip=gclip)
        reset!(f; keepstate=true)
    end

    return sumloss/N
end

# one epoch training
function train(f, data, voc)
    sumloss, numloss = 0.0, 0
    for s in data
        sumloss += iter(f,s)
        numloss += 1
        if numloss % 1000 == 0
            println(numloss)
        end
    end
end

# mean loss
function test(f, data, voc)
    sumloss, numloss = 0.0, 0
    for s in data
        sumloss += iter(f,s;test=true)
        numloss += 1
        if numloss % 1000 == 0
            println(numloss)
        end
    end
    return sumloss/numloss
end
