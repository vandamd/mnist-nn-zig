const std = @import("std");

const Matrix = @import("matrix.zig").Matrix;
const operations = @import("operations.zig");
const activation = @import("activation.zig");

const ActivationFn = *const fn (f64) f64;

pub const Neuron = struct {
    weights: Matrix,
    bias: f64,
    activation: ActivationFn,
    prng: std.rand.DefaultPrng,

    pub fn init(allocator: std.mem.Allocator, numInputs: usize, activationFunction: ActivationFn) !Neuron {
        const weights = try Matrix.init(allocator, 1, numInputs);

        var prng = std.rand.DefaultPrng.init(@as(u64, @intCast(std.time.timestamp())));
        const random = prng.random();
        const bias = random.float(f64);

        for (0..numInputs) |i| {
            weights.data[i] = random.float(f64) * 2 - 1;
        }

        return Neuron{
            .weights = weights,
            .bias = bias,
            .activation = activationFunction,
            .prng = prng,
        };
    }

    pub fn deinit(self: *Neuron, allocator: std.mem.Allocator) void {
        self.weights.deinit(allocator);
    }

    pub fn forward(self: *const Neuron, allocator: std.mem.Allocator, inputs: Matrix) !f64 {
        var inputsTransposed = try operations.transpose(allocator, inputs);
        defer inputsTransposed.deinit(allocator);

        var mult = try operations.multiply(allocator, self.weights, inputsTransposed);
        defer mult.deinit(allocator);

        const sum: f64 = try mult.get(0, 0);
        const value = self.activation(sum + self.bias);
        return value;
    }
};

test "neuron initialisation" {
    const allocator = std.testing.allocator;

    var neuron = try Neuron.init(allocator, 3, activation.relu);
    defer neuron.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), neuron.weights.rows);
    try std.testing.expectEqual(@as(usize, 3), neuron.weights.columns);

    for (neuron.weights.data) |weight| {
        try std.testing.expect(weight >= -1 and weight <= 1);
    }
    try std.testing.expect(neuron.bias >= -1 and neuron.bias <= 1);
}

test "neuron forward" {
    const allocator = std.testing.allocator;

    var neuron = try Neuron.init(allocator, 1, activation.relu);
    defer neuron.deinit(allocator);

    var input = try Matrix.init(allocator, 1, 1);
    defer input.deinit(allocator);
    try input.set(0, 0, 2.0);

    neuron.weights.data[0] = 0.5;
    neuron.bias = 1.0;

    const result = try neuron.forward(allocator, input);
    try std.testing.expectApproxEqAbs(result, 2.0, 0.001);
}
