function bleu(hyp, ref)
    N = 4
    hypothesis_length = 0
    references_length = 0
    clipped = Array(Float64, N)
    total = Array(Float64, N)

    # convert single translation to array
    # works for both single and corpus translation
    hcopy, rcopy = copy(hyp), copy(ref)
    if !isa(hyp, Array)
        hcopy = [hcopy]
        rcopy = isa(ref, Array) ? [rcopy] : [[rcopy]]
    end
    length(hcopy) == length(rcopy) || error("dimensions must match")
    translations = zip(hcopy, rcopy)

    for (hypothesis, references) in translations
        # word count dicts
        hypcounts = map(n -> ngram_counts(hypothesis,n), 1:N)
        refscounts = map(r -> map(n -> ngram_counts(r,n), 1:N), references)

        # corpus length
        hyplen = sum(values(hypcounts[1]))
        reflens = map(x->sum(values(x[1])), refscounts)
        differences = sort(map(x -> (abs(x - hyplen), x), reflens))
        hypothesis_length += hyplen
        references_length += differences[1][2]

        for n = 1:N
            c1 = collect(values(hypcounts[n]))
            c2 = map(
                x -> mapreduce(r -> get(r[n], x, 0), max, refscounts),
                keys(hypcounts[n]))
            clipped[n] += sum(map(k->min(c1[k],c2[k]), 1:length(c1)))
            total[n] += sum(c1)
        end
    end

    scores = zeros(N)
    if length(translations) != 0
        scores = map(i -> clipped[i]/total[i], 1:N)
    end

    brevity_penalty = 1
    if hypothesis_length < references_length
        brevity_penalty = exp(1-references_length/hypothesis_length)
    end
    geometric_mean = reduce(*, scores)^(1/length(scores))
    score = geometric_mean * brevity_penalty

    (score, scores, brevity_penalty, hypothesis_length, references_length)
end

function countdict(xs)
    counts = Dict()
    for x in xs
        if haskey(counts, x)
            counts[x] += 1
        else
            counts[x] = 1
        end
    end
    return counts
end

get_words(sentence) = map(lowercase, split(sentence, " "))
build_ngram(words, n) = map(i -> join(words[i:i+n-1], " "), 1:length(words)-n+1)
ngram_counts(sentence, n) = countdict(build_ngram(get_words(sentence), n))
