using Documenter, PWS

makedocs(
    sitename="PWS.jl",
    pages = [
        "index.md",
        "guide.md",
        "thermodynamic_integration.md"
    ]
)

deploydocs(
    repo = "github.com/manuel-rhdt/PWS.jl.git",
)