SOS = "#SOS#" # start of sentence token
EOS = "#EOS#" # end of sentence token 
PAD = "#PAD#" # padding token
UNK = "#UNK#" # token for unknown words

type Vocabulary
    counts # word counts dict
    sorted # sorted word counts tuple, for stats
    w2i # word to index dict
    i2w # index to word array
    size # vocabulary size, total different words count

    function Vocabulary(words::Array{Any,1}, MIN_OCCUR=5)
        # get word counts
        counts = Dict()
        for word in words
            if haskey(counts, word)
                counts[word] += 1
            else
                counts[word] = 1
            end
        end

        # filter less occured words, build word2index dict upon that collection
        counts = filter((w,o) -> o >= MIN_OCCUR , counts)
        sorted = sort(collect(counts), by = tuple -> last(tuple), rev=true)
        w2i = Dict(SOS => 1)

        i = 2
        for (w,o) in sorted
            w2i[w] = i
            i += 1
        end

        w2i[EOS] = i
        w2i[PAD] = i+1
        w2i[UNK] = i+2

        # let's build index2word array
        i2w = map(j -> "", zeros(i+2))
        for (k,v) in w2i
            i2w[v] = k
        end

        new(counts, sorted, w2i, i2w, i+2)
    end
end

word2index(voc::Vocabulary, w) = haskey(voc.w2i, w) ? voc.w2i[w] : voc.w2i[UNK]
index2word(voc::Vocabulary, i) = voc.i2w[i]
most_occurs(voc::Vocabulary, N) = map(x -> (x.first, y.first), voc.sorted[1:N])
word2onehot(voc::Vocabulary, w) = (v = zeros(Float32,voc.size,1); v[word2index(voc, w)] = 1; v)
sen2vec(voc::Vocabulary, s) = mapreduce(w -> word2index(voc, w), vcat, vcat(SOS, s, EOS))
word2svec(voc::Vocabulary, w) = (v = map(Float32,spzeros(voc.size,1)); v[word2index(voc, w)] = 1; v)
sen2smat(voc::Vocabulary, s) = mapreduce(w -> word2svec(voc, w), hcat, [SOS;s;EOS])
pad2index(voc::Vocabulary) = word2index(voc, PAD)
pad2oneheot(voc::Vocabulary) = word2onehot(voc, PAD) 
