package numerical_algebra

// in-place operations

sub :: proc {
	sub_mat_alloc,
	sub_scalar_alloc,
}


sub_mat_alloc :: proc(a, b: Matrix, allocator := context.allocator) -> (res: Matrix, err: Matrix_Error) {
	if a.rows != b.rows {
		return Matrix{}, Dimension_Mismatch{}
	}
	if a.cols != b.cols {
		return Matrix{}, Dimension_Mismatch{}
	}

	res = alloc(a.rows, a.cols, allocator) or_return


	for i in 0 ..< len(a.data) {
		res.data[i] = a.data[i] - b.data[i]
	}
	return res, nil
}

sub_scalar_alloc :: proc(a: Matrix, s: f32, allocator := context.allocator) -> (res: Matrix, err: Matrix_Error) {
	res = alloc(a.rows, a.cols, allocator) or_return

	for i in 0 ..< len(a.data) {
		res.data[i] = a.data[i] - s
	}

	return res, nil
}
