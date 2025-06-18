using F2Algebra
using Test
using LinearAlgebra

@testset "F2Algebra Tests" begin

    # Test 1: Identity matrix remains unchanged
    @testset "Identity Matrix" begin
        A = bit_eye(3)
        result = rref(A)
        @test result.R == A
        @test result.rank == 3
        @test result.pivots == [1, 2, 3]
    end

    # Test 2: Zero matrix
    @testset "Zero Matrix" begin
        A = falses(3, 3)
        result = rref(A)
        @test result.R == A
        @test result.rank == 0
        @test result.pivots == []
    end

    # Test 3: Full-rank matrix
    @testset "Full-rank Matrix" begin
        A = BitMatrix([1 1 0; 1 0 1; 0 1 0])
        expected = bit_eye(3)
        result = rref(A)
        @test result.R == expected
        @test result.rank == 3
        @test result.pivots == [1, 2, 3]
    end

    # Test 4: Singular matrix
    @testset "Singular Matrix" begin
        A = BitMatrix([1 0 1; 1 0 1; 0 0 0])
        expected = BitMatrix([1 0 1; 0 0 0; 0 0 0])
        result = rref(A)
        @test result.R == expected
        @test result.rank == 1
        @test result.pivots == [1]
    end

    # Test 5: Tall matrix (more rows than columns)
    @testset "Tall Matrix" begin
        A = BitMatrix([1 0; 1 1; 0 1])
        expected = BitMatrix([1 0; 0 1; 0 0])
        result = rref(A)
        @test result.R == expected
        @test result.rank == 2
        @test result.pivots == [1, 2]
    end

    # Test 6: Wide matrix (more columns than rows)
    @testset "Wide Matrix" begin
        A = BitMatrix([1 1 0 1; 0 1 1 0])
        expected = BitMatrix([1 0 1 1; 0 1 1 0])
        result = rref(A)
        @test result.R == expected
        @test result.rank == 2
        @test result.pivots == [1, 2]
    end

    # Test 7: Kernel consistency
    @testset "Kernel Consistency" begin
        A = BitMatrix([1 1 0; 0 1 1])
        result = rref(A)
        null_vec = BitVector([1, 1, 1])
        @test f2_mul(A, null_vec) == BitVector([0, 0])
        @test f2_mul(result.R, null_vec) == BitVector([0, 0])
    end

    # Test 8: In-place rref!
    @testset "In-place rref!" begin
        A = BitMatrix([1 1 0; 1 0 1; 0 1 0])
        A_copy = copy(A)
        result = rref!(A)
        @test A == bit_eye(3)
        @test A != A_copy
        @test result.rank == 3
        @test result.pivots == [1, 2, 3]
    end
end
