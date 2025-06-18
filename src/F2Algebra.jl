module F2Algebra

export rref!, rref, bit_eye, f2_mul

"""
    bit_eye(n::Integer) -> BitMatrix

Construct the `n × n` identity matrix over 𝔽₂, represented as a `BitMatrix`.
"""
function bit_eye(n::Integer)
    A = falses(n, n)
    for i in 1:n
        A[i, i] = true
    end
    return A
end

"""
    f2_mul(A::AbstractMatrix{Bool}, x::AbstractVector{Bool})

Compute the matrix-vector product `A * x` over 𝔽₂ (mod 2).

Returns a `BitVector` of length `size(A, 1)`, where each entry is computed
as the XOR dot product of a row of `A` with `x`.
"""
function f2_mul(A::AbstractMatrix{Bool}, x::AbstractVector{Bool})
    m, n = size(A)
    if length(x) != n
        throw(DimensionMismatch("length of vector must match number of columns"))
    end
    y = falses(m)
    for i in 1:m
        acc = false
        for j in 1:n
            acc ⊻= A[i, j] & x[j]
        end
        y[i] = acc
    end
    return y
end

"""
    rref!(A::BitMatrix) -> (; R, rank, pivots)

Compute the Reduced Row Echelon Form (RREF) of a `BitMatrix` in-place
using Gaussian elimination over 𝔽₂.

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

        # Eliminate all other entries in column j
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

Non-mutating version of `rref!`, returns a named tuple with a fresh copy.
"""
function rref(A::BitMatrix)
    return rref!(copy(A))
end

end # module F2Algebra
