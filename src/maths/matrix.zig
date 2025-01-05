const std = @import("std");
const testing = std.testing;

pub const MatrixError = error{
    RowOutOfBounds,
    ColumnOutOfBounds,
};

pub const Matrix = struct {
    rows: usize,
    columns: usize,
    data: []f64,

    pub fn init(allocator: std.mem.Allocator, rows: usize, columns: usize) !Matrix {
        const data = try allocator.alloc(f64, rows * columns);

        return Matrix{
            .rows = rows,
            .columns = columns,
            .data = data,
        };
    }

    pub fn deinit(self: *Matrix, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn get(self: Matrix, row: usize, column: usize) MatrixError!f64 {
        if (row >= self.rows) return MatrixError.RowOutOfBounds;
        if (column >= self.columns) return MatrixError.ColumnOutOfBounds;

        return self.data[(row * self.columns) + column];
    }

    pub fn set(self: *Matrix, row: usize, column: usize, value: f64) MatrixError!void {
        if (row >= self.rows) return MatrixError.RowOutOfBounds;
        if (column >= self.columns) return MatrixError.ColumnOutOfBounds;

        self.data[(row * self.columns) + column] = value;
    }
};

test "matrix creation" {
    const allocator = testing.allocator;

    var matrix = try Matrix.init(allocator, 3, 5);
    defer matrix.deinit(allocator);

    try testing.expectEqual(@as(usize, 3), matrix.rows);
    try testing.expectEqual(@as(usize, 5), matrix.columns);
}

test "matrix get and set" {
    const allocator = testing.allocator;

    var matrix = try Matrix.init(allocator, 2, 2);
    defer matrix.deinit(allocator);

    try matrix.set(0, 1, 5);
    const value = matrix.get(0, 1);

    try testing.expectEqual(@as(f64, 5), value);
}

test "matrix bounds checker" {
    const allocator = testing.allocator;

    var matrix = try Matrix.init(allocator, 2, 5);
    defer matrix.deinit(allocator);

    try testing.expectError(MatrixError.RowOutOfBounds, matrix.get(5, 2));
    try testing.expectError(MatrixError.ColumnOutOfBounds, matrix.get(1, 20));
}
