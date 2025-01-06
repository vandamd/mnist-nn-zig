const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;

pub const ActivationFn = *const fn (f64) f64;

pub fn relu(x: f64) f64 {
    return @max(0, x);
}

pub fn reluDerivative(x: f64) f64 {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + std.math.exp(-x));
}

pub fn sigmoidDerivative(x: f64) f64 {
    return sigmoid(x) * (1 - sigmoid(x));
}

pub fn softmax(allocator: std.mem.Allocator, input: Matrix) !Matrix {
    var result = try Matrix.init(allocator, 1, input.columns);

    var max: f64 = -std.math.inf(f64);
    for (0..input.columns) |i| {
        max = @max(max, try input.get(0, i));
    }

    var sum: f64 = 0;
    for (0..input.columns) |i| {
        const value = @exp(try input.get(0, i) - max); // Subtract max for stability
        try result.set(0, i, value);
        sum += value;
    }

    for (0..result.columns) |i| {
        try result.set(0, i, (try result.get(0, i)) / sum);
    }

    return result;
}

test "softmax" {
    const allocator = std.testing.allocator;

    var input = try Matrix.init(allocator, 1, 2);
    defer input.deinit(allocator);

    try input.set(0, 0, 1.0);
    try input.set(0, 1, 2.0);

    var result = try softmax(allocator, input);
    defer result.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), result.rows);
    try std.testing.expectEqual(@as(usize, 2), result.columns);

    var sum: f64 = 0;
    for (0..result.columns) |i| {
        const value = try result.get(0, i);
        try std.testing.expect(value >= 0 and value <= 1);
        sum += value;
    }

    try std.testing.expectApproxEqAbs(sum, 1.0, 0.001);

    try std.testing.expect(try result.get(0, 1) > try result.get(0, 0));
}
