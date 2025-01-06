const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

pub const MatrixOpError = error{
    DimensionMismatch,
    OutOfMemory,
};

pub fn add(allocator: std.mem.Allocator, a: Matrix, b: Matrix) MatrixOpError!Matrix {
    if (a.rows != b.rows or a.columns != b.columns) {
        return MatrixOpError.DimensionMismatch;
    }

    var result = try Matrix.init(allocator, a.rows, a.columns);

    for (0..a.rows * a.columns) |i| {
        result.data[i] = a.data[i] + b.data[i];
    }

    return result;
}

pub fn multiply(allocator: std.mem.Allocator, a: Matrix, b: Matrix) (MatrixOpError || Matrix.Error)!Matrix {
    if (a.columns != b.rows) {
        return MatrixOpError.DimensionMismatch;
    }

    var result = try Matrix.init(allocator, a.rows, b.columns);

    for (0..a.rows) |row| {
        for (0..b.columns) |col| {
            var sum: f64 = 0;

            for (0..a.columns) |k| {
                const elementA = try a.get(row, k);
                const elementB = try b.get(k, col);
                sum += elementA * elementB;
            }

            try result.set(row, col, sum);
        }
    }

    return result;
}

pub fn transpose(allocator: std.mem.Allocator, m: Matrix) MatrixOpError!Matrix {
    var result = try Matrix.init(allocator, m.columns, m.rows);

    for (0..m.rows) |i| {
        for (0..m.columns) |j| {
            const origIdx = (i * m.columns) + j;
            const newIdx = (j * m.rows) + i;
            result.data[newIdx] = m.data[origIdx];
        }
    }

    return result;
}

test "matrix addition" {
    const allocator = std.testing.allocator;

    var matrix1 = try Matrix.init(allocator, 2, 2);
    defer matrix1.deinit(allocator);
    var matrix2 = try Matrix.init(allocator, 2, 2);
    defer matrix2.deinit(allocator);

    const testData = [_]f64{1.0} ** 4;
    const expectedData = [_]f64{2.0} ** 4;

    @memcpy(matrix1.data, &testData);
    @memcpy(matrix2.data, &testData);

    var matrix3 = try add(allocator, matrix1, matrix2);
    defer matrix3.deinit(allocator);

    try std.testing.expectEqualSlices(f64, &expectedData, matrix3.data);
}

test "matrix addition dimension checker" {
    const allocator = std.testing.allocator;

    var matrix1 = try Matrix.init(allocator, 2, 2);
    defer matrix1.deinit(allocator);
    var matrix2 = try Matrix.init(allocator, 4, 2);
    defer matrix2.deinit(allocator);

    try std.testing.expectError(MatrixOpError.DimensionMismatch, add(allocator, matrix1, matrix2));
}

test "matrix multiplication" {
    const allocator = std.testing.allocator;

    var matrix1 = try Matrix.init(allocator, 2, 2);
    defer matrix1.deinit(allocator);
    var matrix2 = try Matrix.init(allocator, 2, 1);
    defer matrix2.deinit(allocator);

    const testData1 = [_]f64{ 2, 2, 1, 1 };
    const testData2 = [_]f64{ 1, 2 };

    const expectedData = [_]f64{ 6, 3 };

    @memcpy(matrix1.data, &testData1);
    @memcpy(matrix2.data, &testData2);

    var matrix3 = try multiply(allocator, matrix1, matrix2);
    defer matrix3.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), matrix3.rows);
    try std.testing.expectEqual(@as(usize, 1), matrix3.columns);
    try std.testing.expectEqualSlices(f64, &expectedData, matrix3.data);
}

test "matrix multiplication dimension checker" {
    const allocator = std.testing.allocator;

    var matrix1 = try Matrix.init(allocator, 2, 1);
    defer matrix1.deinit(allocator);
    var matrix2 = try Matrix.init(allocator, 2, 2);
    defer matrix2.deinit(allocator);

    try std.testing.expectError(MatrixOpError.DimensionMismatch, multiply(allocator, matrix1, matrix2));
}

test "matrix transpose" {
    const allocator = std.testing.allocator;

    var matrix = try Matrix.init(allocator, 2, 3);
    defer matrix.deinit(allocator);

    const testData = [_]f64{ 1, 2, 3, 4, 5, 6 };
    const expectedData = [_]f64{ 1, 4, 2, 5, 3, 6 };

    @memcpy(matrix.data, &testData);

    var transposed = try transpose(allocator, matrix);
    defer transposed.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), transposed.rows);
    try std.testing.expectEqual(@as(usize, 2), transposed.columns);
    try std.testing.expectEqualSlices(f64, &expectedData, transposed.data);
}
