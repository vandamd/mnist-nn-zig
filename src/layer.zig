const std = @import("std");

const Neuron = @import("neuron.zig").Neuron;
const Matrix = @import("matrix.zig").Matrix;
const operations = @import("operations.zig");
const activation = @import("activation.zig");

pub const Layer = struct {
    neurons: []Neuron,
    numNeurons: usize,
    numInputs: usize,
    activationFunction: activation.ActivationFn,

    pub fn init(allocator: std.mem.Allocator, numNeurons: usize, numInputs: usize, activationFunction: activation.ActivationFn) !Layer {
        var neurons = try allocator.alloc(Neuron, numNeurons);

        for (0..numNeurons) |i| {
            neurons[i] = try Neuron.init(allocator, numInputs, activationFunction);
        }

        return Layer{
            .neurons = neurons,
            .numNeurons = numNeurons,
            .numInputs = numInputs,
            .activationFunction = activationFunction,
        };
    }

    pub fn deinit(self: *Layer, allocator: std.mem.Allocator) void {
        for (0..self.numNeurons) |i| {
            self.neurons[i].deinit(allocator);
        }
        allocator.free(self.neurons);
    }

    pub fn forward(self: *const Layer, allocator: std.mem.Allocator, inputs: Matrix) !Matrix {
        var outputs = try Matrix.init(allocator, 1, self.numNeurons);

        for (0..self.numNeurons) |i| {
            try outputs.set(0, i, try self.neurons[i].forward(allocator, inputs));
        }

        return outputs;
    }
};

test "neuron layer init and deinit" {
    const allocator = std.testing.allocator;

    var layer = try Layer.init(allocator, 2, 3, activation.relu);
    defer layer.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), layer.numNeurons);
    try std.testing.expectEqual(@as(usize, 3), layer.numInputs);
    try std.testing.expectEqual(@as(usize, 2), layer.neurons.len);

    for (layer.neurons) |neuron| {
        try std.testing.expectEqual(@as(usize, 1), neuron.weights.rows);
        try std.testing.expectEqual(@as(usize, 3), neuron.weights.columns);

        for (neuron.weights.data) |weight| {
            try std.testing.expect(weight >= -1 and weight <= 1);
        }

        try std.testing.expect(neuron.bias >= -1 and neuron.bias <= 1);
    }
}

test "layer forward" {
    const allocator = std.testing.allocator;

    var layer = try Layer.init(allocator, 2, 3, activation.relu);
    defer layer.deinit(allocator);

    var input = try Matrix.init(allocator, 1, 3);
    defer input.deinit(allocator);

    for (0..layer.numNeurons) |i| {
        for (0..layer.numInputs) |j| {
            try input.set(0, j, 2.0);
            layer.neurons[i].weights.data[j] = 2.0;
        }
        layer.neurons[i].bias = 1;
    }

    var results = try layer.forward(allocator, input);
    defer results.deinit(allocator);

    for (0..results.columns) |i| {
        try std.testing.expectApproxEqAbs(try results.get(0, i), 13, 0.001);
    }

    try std.testing.expectEqual(@as(usize, 1), results.rows);
    try std.testing.expectEqual(@as(usize, 2), results.columns);
}
