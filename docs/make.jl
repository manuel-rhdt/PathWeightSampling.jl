using Documenter, PathWeightSampling

makedocs(
    sitename="PathWeightSampling.jl",
    pages = [
        "index.md",
        "guide.md",
        "systems.md",
        "examples.md",
        "marginalization.md",
        "write_output.md"
    ]
)

deploydocs(
    repo = "github.com/manuel-rhdt/PathWeightSampling.jl.git",
)