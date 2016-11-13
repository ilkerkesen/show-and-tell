using ArgParse, JLD, JSON

function main(args=ARGS)
    s = ArgParseSettings()
    s.description = string(
        "Show and Tell: A Neural Image Caption Generator",
        " Knet implementation by Ilker Kesen [ikesen16_at_ku.edu.tr], 2016.")

    @add_arg_table s begin
        ("--features"; help="features file")
        ("--captions"; help="captions file")
        ("--savefile"; help="output file")
    end

    features = 
    
end

!isinteractive() && !isdefined(Core.Main, :load_only) && main(ARGS)
