const std = @import("std");
const Matrix = @import("matrix.zig").Matrix;
pub const operations = @import("operations.zig");

pub fn mse(allocator: std.mem.Allocator, prediction: Matrix, target: Matrix) !f64 {
    var diff = try operations.subtract(allocator, prediction, target);
    defer diff.deinit(allocator);

    var sum: f64 = 0;
    for (0..diff.columns) |i| {
        const value = try diff.get(0, i);
        sum += value * value;
    }

    return sum / @as(f64, @floatFromInt(diff.columns));
}

pub fn mseGradient(allocator: std.mem.Allocator, prediction: Matrix, target: Matrix) !Matrix {
    var result = try Matrix.init(allocator, 1, prediction.columns);
    const n = @as(f64, @floatFromInt(prediction.columns));

    for (0..result.columns) |i| {
        const diff = 2 * ((try prediction.get(0, i)) - (try target.get(0, i)));
        try result.set(0, i, diff / n);
    }

    return result;
}

pub fn crossEntropyLoss(prediction: Matrix, target: Matrix) !f64 {
    const epsilon = 1e-15;
    var sum: f64 = 0;

    for (0..prediction.columns) |i| {
        const pred = try prediction.get(0, i);
        const targ = try target.get(0, i);
        sum += targ * @log(pred + epsilon);
    }

    return -sum;
}

pub fn crossEntropyGradient(allocator: std.mem.Allocator, prediction: Matrix, target: Matrix) !Matrix {
    const epsilon = 1e-15;
    var result = try Matrix.init(allocator, 1, prediction.columns);

    for (0..result.columns) |i| {
        const pred = try prediction.get(0, i) + epsilon;
        try result.set(0, i, -1 * try target.get(0, i) / pred);
    }

    return result;
}

test "mse" {
    const allocator = std.testing.allocator;

    var prediction = try Matrix.init(allocator, 1, 3);
    defer prediction.deinit(allocator);

    var trueLabels = try Matrix.init(allocator, 1, 3);
    defer trueLabels.deinit(allocator);

    const predictionData = [_]f64{ 0.8, 0.1, 0.1 };
    const trueLabelsData = [_]f64{ 1.0, 0.0, 0.0 };

    @memcpy(prediction.data, &predictionData);
    @memcpy(trueLabels.data, &trueLabelsData);

    const result = try mse(allocator, prediction, trueLabels);

    try std.testing.expectApproxEqAbs(0.02, result, 0.001);
}

test "cross entropy loss" {
    const allocator = std.testing.allocator;

    var prediction = try Matrix.init(allocator, 1, 3);
    defer prediction.deinit(allocator);

    var trueLabels = try Matrix.init(allocator, 1, 3);
    defer trueLabels.deinit(allocator);

    const predictionData = [_]f64{ 0.7, 0.2, 0.1 };
    const trueLabelsData = [_]f64{ 0.0, 1.0, 0.0 };

    @memcpy(prediction.data, &predictionData);
    @memcpy(trueLabels.data, &trueLabelsData);

    const result = try crossEntropyLoss(prediction, trueLabels);

    try std.testing.expectApproxEqAbs(1.609, result, 0.001);
}

test "mse gradient" {
    const allocator = std.testing.allocator;

    var prediction = try Matrix.init(allocator, 1, 3);
    defer prediction.deinit(allocator);

    var target = try Matrix.init(allocator, 1, 3);
    defer target.deinit(allocator);

    const predictionData = [_]f64{ 0.8, 0.1, 0.1 };
    const targetData = [_]f64{ 1.0, 0.0, 0.0 };

    @memcpy(prediction.data, &predictionData);
    @memcpy(target.data, &targetData);

    var gradient = try mseGradient(allocator, prediction, target);
    defer gradient.deinit(allocator);

    // For first value: 2(0.8 - 1.0)/3 = -0.133
    try std.testing.expectApproxEqAbs(try gradient.get(0, 0), -0.133, 0.001);
    // For second value: 2(0.1 - 0.0)/3 = 0.067
    try std.testing.expectApproxEqAbs(try gradient.get(0, 1), 0.067, 0.001);
}

test "cross entropy gradient" {
    const allocator = std.testing.allocator;

    var prediction = try Matrix.init(allocator, 1, 3);
    defer prediction.deinit(allocator);

    var target = try Matrix.init(allocator, 1, 3);
    defer target.deinit(allocator);

    const predictionData = [_]f64{ 0.7, 0.2, 0.1 };
    const targetData = [_]f64{ 0.0, 1.0, 0.0 };

    @memcpy(prediction.data, &predictionData);
    @memcpy(target.data, &targetData);

    var gradient = try crossEntropyGradient(allocator, prediction, target);
    defer gradient.deinit(allocator);

    // For first value: -0/0.7 = 0
    try std.testing.expectApproxEqAbs(try gradient.get(0, 0), 0.0, 0.001);
    // For second value: -1/0.2 = -5
    try std.testing.expectApproxEqAbs(try gradient.get(0, 1), -5.0, 0.001);
}
