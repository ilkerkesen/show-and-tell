using ArgParse
using JLD
using MAT

function main(args)
    s = ArgParseSettings()
    s.description = "Caption generation script for the model (raw images)"

    @add_arg_table s begin
        ("--filenames"; help="filenames file")
        ("--generations"; help="hypothesis file")
        ("--references"; nargs='+'; help="references files")
    end

    # parse args
    println("Datetime: ", now()); flush(STDOUT)
    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true); println(o)

    # read files
    filenames = readfile(o[:filenames])
    generations = readfile(o[:generations])
    if length(filenames) != length(generations)
        error("filenames length mismatch")
    end

    references = Array(Any, length(o[:references]))
    for k = 1:length(o[:references])
        references[k] = readfile(o[:references][k])
        if length(references[k]) != length(generations)
            error("references length mismatch")
        end
    end

    translations = []
    for k = 1:length(generations)
        entry = Dict(
            "filename" => filenames[k],
            "hypothesis" => generations[k],
            "references" => map(r -> r[k], references))
        push!(translations, entry)
    end

    # evaluate
    ti = now()
    @printf("Evaluation started (date=%s)\n", ti)
    score, scores, bp, srclen, tarlen = bleu(translations)
    @printf("BLEU = %.1f, %.1f/%.1f/%.1f/%.1f ",
            100*score, map(i->i*100,scores)...)
    @printf("(BP=%g, ratio=%g, hyp_len=%d, ref_len=%d)\n",
            bp, srclen/tarlen, srclen, tarlen)
    tf = now()
    @printf("\nTime elapsed: %s [%s]\n", tf-ti, tf)
end

function meteor(translations)
    precision = 0
    recall = 0
    penalty = 0
    for translation in translations

    end
end

function meteor(hypothesis, reference)
    hypwords = get_words(hypothesis)
    refwords = get_words(reference)

    precision = reduce((acc,x) -> in(x,refwords) + acc, 0, hypwords)
    recall    = reduce((acc,x) -> in(x,hypwords) + acc, 0, refwords)
    fscore    = 0
end

function get_chunk_count(hyp, ref)
    similiarity(x1,x2) = Int(x1==x2)-1
    hypwords = get_words(hyp)
    refwords = get_words(ref)

    # init alignment matrix
    align = zeros(length(refwords)+1, length(hypwords)+1)
    for j = 2:size(align,2)
        align[0,j] = 1-j
    end
    for i = 2:size(align,1)
        align[i,0] = 1-i
    end

    # calculate alignment matrix
    for i = 2:size(align,1)
        for j = 2:size(align,2)
            p1 = simila
        end
    end
end


function bleu(translations)
    N = 4
    hypothesis_length = 0
    references_length = 0
    clipped = Array(Float64, N)
    total = Array(Float64, N)
    for translation in translations
        hypothesis = translation["hypothesis"]
        references = translation["references"]

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

function readfile(filename)
    f = open(abspath(filename), "r")
    txt = split(readstring(f), "\n")
    txt[end] == "" && pop!(txt)
    close(f)
    return txt
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
