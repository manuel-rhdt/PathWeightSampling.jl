using Documenter, PathWeightSampling

makedocs(
    sitename="PathWeightSampling.jl",
    pages = [
        "Home" => "index.md",
        "Getting Started" => "guide.md",
        "Tutorials" => [
            "Three-Species Cascade" => "three_species_tutorial.md",
            "Hybrid Continuous-Discrete System" => "hybrid_system_tutorial.md"
        ],
        "Theory" => [
            "Marginalization Strategies" => "marginalization.md"
        ],
        "Reference" => [
            "System Types" => "systems.md",
            "Output & Saving" => "write_output.md",
            "Examples" => "examples.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/manuel-rhdt/PathWeightSampling.jl.git",
)