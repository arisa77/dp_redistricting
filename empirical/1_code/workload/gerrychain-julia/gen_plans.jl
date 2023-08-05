using GerryChain
using Shapefile
using JSON
#using DataFrames

# run: julia --project=. gen_plans.jl

function get_params(
    state::String,
    )

    params = Dict{String,Any}()

    if state == "IA"
        params["state"] = state
        params["pipeline"] = "../../2_pipeline/gerrychain-julia/gen_plans/IA/"
        params["shapefile"] = "../../0_data/external/IA-shapefiles/IA_counties/IA_counties.shp"
        params["population col"] = "TOTPOP"
        params["assignment col"] = "CD"
        params["popbound"] = 0.1
        params["numsteps"] = 10000
        params["election names"] = Dict("name" => "PRES16", "dname" => "PRES16D", "rname" => "PRES16R")
        params["usedpcounts"] = true

        if params["usedpcounts"] == true
            params["strategy"] = "Identity"
            params["post_process"] = "ls-round"
            params["epsilon"] = "1e-06"#"0.001"
            params["instance"] = 1 #first instance
            params["randomseedfile"] = "../../2_pipeline/redistricting/generate_random_seed/store/random_seed_IA.json"
            params["scorefile"] =  string(params["pipeline"],"scores_compact_pop",params["popbound"],"_eps",params["epsilon"],"_",params["strategy"],"_",params["post_process"],".json")
            params["assignmentfile"] =  string(params["pipeline"],"assignment_compact_pop",params["popbound"],"_eps",params["epsilon"],"_",params["strategy"],"_",params["post_process"],".json")
        else
            #params["scorefile"] = params["pipeline"]*"scores.json"
            params["scorefile"] =  string(params["pipeline"],"scores_compact_pop",params["popbound"],"_gt.json")
            params["assignmentfile"] =  string(params["pipeline"],"assignment_compact_pop",params["popbound"],"_gt.json")
        end 
        

    elseif state == "NC"

        params["state"] = state
        params["pipeline"] = "../../2_pipeline/gerrychain-julia/gen_plans/NC/"
        params["shapefile"] = "../../0_data/external/NC-shapefiles/NC_VTD/NC_VTD.shp"
        params["population col"] = "TOTPOP"
        params["assignment col"] = "newplan"
        params["popbound"] = 0.1
        params["numsteps"] = 10000
        params["election names"] = Dict("name" => "PRES16", "dname" => "EL16G_PR_D", "rname" => "EL16G_PR_R")
        params["usedpcounts"] = true

        if params["usedpcounts"] == true
            params["strategy"] = "Identity"
            params["post_process"] = "ls-round"
            params["epsilon"] = "1e-06"#"0.001"
            params["instance"] = 1 #first instance
            params["randomseedfile"] = string("../../2_pipeline/redistricting/generate_random_seed/store/random_seed_NC","_",params["strategy"],"_",params["post_process"],"_",params["epsilon"],".json")
            params["scorefile"] =  string(params["pipeline"],"scores_compact_pop",params["popbound"],"_eps",params["epsilon"],"_",params["strategy"],"_",params["post_process"],".json")
            params["assignmentfile"] =  string(params["pipeline"],"assignment_compact_pop",params["popbound"],"_eps",params["epsilon"],"_",params["strategy"],"_",params["post_process"],".json")
        else
            params["scorefile"] =  string(params["pipeline"],"scores_compact_pop",params["popbound"],"_gt.json")
            params["assignmentfile"] =  string(params["pipeline"],"assignment_compact_pop",params["popbound"],"_gt.json")
        end 

    elseif state == "MA"

        params["state"] = state
        params["pipeline"] = "../../2_pipeline/gerrychain-julia/gen_plans/MA/"
        params["shapefile"] = "../../0_data/manual/MA-shapefiles/MA_no_islands_12_16/MA_precincts_12_16.shp"
        params["population col"] = "POP10"
        params["assignment col"] = "CD"
        params["popbound"] = 0.1
        params["numsteps"] = 10000
        params["election names"] = Dict("name" => "PRES16", "dname" => "PRES16D", "rname" => "PRES16R")
        params["usedpcounts"] = true

        if params["usedpcounts"] == true
            params["strategy"] = "Identity"
            params["post_process"] = "ls-round"
            params["epsilon"] = "1e-05"#"0.001"
            params["instance"] = 1 #first instance
            params["randomseedfile"] = string("../../2_pipeline/redistricting/generate_random_seed/store/random_seed_MA","_",params["strategy"],"_",params["post_process"],"_",params["epsilon"],".json")
            params["scorefile"] =  string(params["pipeline"],"scores_compact_pop",params["popbound"],"_eps",params["epsilon"],"_",params["strategy"],"_",params["post_process"],".json")
            params["assignmentfile"] =  string(params["pipeline"],"assignment_compact_pop",params["popbound"],"_eps",params["epsilon"],"_",params["strategy"],"_",params["post_process"],".json")
        else
            params["scorefile"] =  string(params["pipeline"],"scores_compact_pop",params["popbound"],"_gt.json")
            params["assignmentfile"] =  string(params["pipeline"],"assignment_compact_pop",params["popbound"],"_gt.json")
        end

    elseif state == "CT"

        params["state"] = state
        params["pipeline"] = "../../2_pipeline/gerrychain-julia/gen_plans/CT/"
        params["shapefile"] = "../../0_data/external/CT-shapefiles/CT_precincts/CT_precincts.shp"
        params["population col"] = "TOTPOP"
        params["assignment col"] = "CD"
        params["popbound"] = 0.1
        params["numsteps"] = 10000
        params["election names"] = Dict("name" => "PRES18", "dname" => "USH18D", "rname" => "USH18R")
        params["usedpcounts"] = true

        if params["usedpcounts"] == true
            params["strategy"] = "Identity"
            params["post_process"] = "ls-round"
            params["epsilon"] = "1e-05"#"0.001"
            params["instance"] = 1 #first instance
            params["randomseedfile"] = string("../../2_pipeline/redistricting/generate_random_seed/store/random_seed_CT","_",params["strategy"],"_",params["post_process"],"_",params["epsilon"],".json")
            params["scorefile"] =  string(params["pipeline"],"scores_compact_pop",params["popbound"],"_eps",params["epsilon"],"_",params["strategy"],"_",params["post_process"],".json")
            params["assignmentfile"] =  string(params["pipeline"],"assignment_compact_pop",params["popbound"],"_eps",params["epsilon"],"_",params["strategy"],"_",params["post_process"],".json")
        else
            params["scorefile"] =  string(params["pipeline"],"scores_compact_pop",params["popbound"],"_gt.json")
            params["assignmentfile"] =  string(params["pipeline"],"assignment_compact_pop",params["popbound"],"_gt.json")
        end
    end



    return params

end

function check_initial_seed(
    graph::BaseGraph,
    initial_partition::Partition,
    pop_bound::Float64,
    )
    #check if initial_partition is valid

    ideal_pop = graph.total_pop/initial_partition.num_dists
    min_pop = Int(ceil((1 - pop_bound) * ideal_pop))
    max_pop = Int(floor((1 + pop_bound) * ideal_pop))
    
    for dist_pop in initial_partition.dist_populations
        if dist_pop < min_pop || dist_pop > max_pop
            println("initial seed is not valid")
            println(initial_partition.dist_populations,min_pop,max_pop,ideal_pop)
            exit()
        end 
    end

end
function run_chain(
    graph::BaseGraph,
    initial_partition::Partition,
    election_names::Dict{String,String},
    num_steps::Int,
    pop_bound::Float64,
    )

    #check if initial_partition is valid

    ideal_pop = graph.total_pop/initial_partition.num_dists
    min_pop = Int(ceil((1 - pop_bound) * ideal_pop))
    max_pop = Int(floor((1 + pop_bound) * ideal_pop))
    
    for dist_pop in initial_partition.dist_populations
        if dist_pop < min_pop || dist_pop > max_pop
            println("initial seed is not valid")
            exit()
        end 
    end


    # Define parameters of chain (number of steps and population constraint)
    pop_constraint = PopulationConstraint(graph, initial_partition, pop_bound)
    compact_constraint = CompactnessConstraint(initial_partition)
    
    # Initialize Election of interest
    election = Election(election_names["name"], [election_names["dname"], election_names["rname"]], initial_partition.num_dists)
    # Define election-related metrics and scores
    election_metrics = [
        vote_count("vote_count", election, election_names["dname"]),
        efficiency_gap("efficiency_gap", election, election_names["dname"]),
        seats_won("seats_won", election, election_names["dname"]),
        mean_median("mean_median", election, election_names["dname"]),
        vote_share("vote_share", election, election_names["dname"])
    ]
    scores = [
            #DistrictAggregate("presd", "PRES16D"),
            ElectionTracker(election, election_metrics),
            num_cut_edges("cut_edges")
    ]

    # Run the chain
    println("Running $num_steps -step ReCom chain...")
    @time chain_score_data, partitions = recom_save_plans(graph, initial_partition, pop_constraint, compact_constraint, num_steps, scores,num_tries=3,no_self_loops=true)
    return chain_score_data, partitions
end
#################################
dic_scores = Dict()
dic_partitions = Dict()

params = get_params("IA")
#params = get_params("NC")
#params = get_params("CT")
#params = get_params("MA")
# Initialize graph 
graph = BaseGraph(params["shapefile"], params["population col"])


if params["usedpcounts"] != true
    #run on ground truth count
    if params["use random gt seed"] == true
        scores,partitions = run_chain(graph,Partition(graph, Int.(JSON.parsefile(params["gtseedfile"])["gt"])), params["election names"], params["gtnumsteps"], params["popbound"])
    else
        scores,partitions = run_chain(graph,Partition(graph, params["assignment col"]), params["election names"], params["gtnumsteps"], params["popbound"])
    end

    dic_scores["gt"] = scores.step_values
    dic_partitions["gt"] = transpose(partitions)

else

    initials = JSON.parsefile(params["randomseedfile"])
    pop_partitions = initials[params["strategy"]][params["post_process"]][params["epsilon"]][params["population col"]]
            
    pop = Int.(pop_partitions[params["instance"]][1])
    initial_assignment = Int.(pop_partitions[params["instance"]][2])
    updatedgraph = update_populations(graph, pop)

    initial_partition = Partition(updatedgraph, initial_assignment)
    #check_initial_seed(updatedgraph,initial_partition,params["popbound"])

    scores,partitions = run_chain(updatedgraph,initial_partition, params["election names"], params["numsteps"], params["popbound"])
        
    dic_scores["dp"] = scores.step_values
    dic_partitions["dp"]= transpose(partitions)
             
end




print("save scores and assignements")
#save score and partitions into json format
open(params["scorefile"], "w") do f
    print(f, JSON.json(dic_scores))
end

open(params["assignmentfile"], "w") do f
    print(f, JSON.json(dic_partitions))
end
