module F2Algebra

export rref!, rref

"""
    rref!(A::BitMatrix) -> (; R, rank, pivots)

Compute the Reduced Row Echelon Form (RREF) of a BitMatrix in-place using Gaussian elimination over F₂.
Returns a named tuple:
  - `R`: the matrix in RREF (same object as `A`),
  - `rank`: number of pivots,
  - `pivots`: list of pivot column indices.
"""
function rref!(A::BitMatrix)
    m, n = size(A)
    r = 1  # current pivot row
    pivots = Int[]

    for j in 1:n
        if r > m
            break
        end

        # Find pivot in column j
        i = findfirst(A[r:end, j])
        if isnothing(i)
            continue
        end
        i += r - 1

        # Swap pivot row with current row
        if i != r
            A[i, :], A[r, :] = A[r, :], A[i, :]
        end

        # Eliminate other entries in column j
        for k in 1:m
            if k != r && A[k, j]
                A[k, :] .⊻= A[r, :]
            end
        end

        push!(pivots, j)
        r += 1
    end

    return (R = A, rank = length(pivots), pivots = pivots)
end

"""
    rref(A::BitMatrix) -> (; R, rank, pivots)

Non-mutating version of `rref!`, returns a named tuple.
"""
rref(A::BitMatrix) = rref!(copy(A))

end # module F2Algebra
