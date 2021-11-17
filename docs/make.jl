using Documenter, PathWeightSampling

makedocs(
    sitename="PathWeightSampling.jl",
    pages = [
        "index.md",
        "guide.md",
        "systems.md",
        "examples.md",
        "marginalization.md",
    ]
)

deploydocs(
    repo = "github.com/manuel-rhdt/PathWeightSampling.jl.git",
)