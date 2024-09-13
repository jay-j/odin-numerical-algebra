package numerical_algebra
import "core:math/rand"
import "core:testing"

// TODO: in-place operations

add :: proc {
	add_mat_alloc,
	add_scalar_alloc,
}


add_mat_alloc :: proc(a, b: Matrix, allocator := context.allocator) -> (res: Matrix, err: Matrix_Error) {
	if a.rows != b.rows {
		return Matrix{}, Dimension_Mismatch{}
	}
	if a.cols != b.cols {
		return Matrix{}, Dimension_Mismatch{}
	}

	res = alloc(a.rows, a.cols, allocator) or_return


	for i in 0 ..< len(a.data) {
		res.data[i] = a.data[i] + b.data[i]
	}
	return res, nil
}

add_scalar_alloc :: proc(a: Matrix, s: f32, allocator := context.allocator) -> (res: Matrix, err: Matrix_Error) {
	res = alloc(a.rows, a.cols, allocator) or_return

	for i in 0 ..< len(a.data) {
		res.data[i] = a.data[i] + s
	}

	return res, nil
}

@(test)
test_add :: proc(t: ^testing.T) {
	rand.reset(0)

	A, _ := alloc(3, 4)
	B, _ := alloc(3, 4)

	fill_random_range(&A, -1, 1, context.random_generator)
	fill_random_range(&B, -1, 1, context.random_generator)

	set(&A, 0, 3, 100)
	set(&B, 2, 1, -100)

	C, err := add(A, B)
	testing.expect_value(t, err, nil)

	testing.expect(t, C.data[5] < -90)
	testing.expect(t, C.data[9] > 90)
	testing.expect_value(t, C.data[0], A.data[0] + B.data[0])
}
