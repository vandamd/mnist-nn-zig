const std = @import("std");

const Layer = @import("layer.zig").Layer;
const Neuron = @import("neuron.zig").Neuron;
const Matrix = @import("matrix.zig").Matrix;
const operations = @import("operations.zig");
const activation = @import("activation.zig");

pub const NetworkError = error{
    tooFewLayers,
};

pub const Network = struct {
    layers: []Layer,
    numLayers: usize,
    neuronConfig: []const u32,

    pub fn init(allocator: std.mem.Allocator, numFeatures: u64, neuronConfig: []const u32, activationFunction: activation.ActivationFn) !Network {
        const numLayers = neuronConfig.len;

        if (numLayers < 2) {
            return NetworkError.tooFewLayers;
        }

        var layers = try allocator.alloc(Layer, numLayers);

        for (0..numLayers) |i| {
            var numInputs: u64 = 0;
            if (i == 0) {
                numInputs = numFeatures;
            } else {
                numInputs = neuronConfig[i - 1];
            }

            layers[i] = try Layer.init(allocator, neuronConfig[i], numInputs, activationFunction);
        }

        return Network{
            .layers = layers,
            .numLayers = numLayers,
            .neuronConfig = neuronConfig,
        };
    }

    pub fn deinit(self: *Network, allocator: std.mem.Allocator) void {
        for (0..self.numLayers) |i| {
            self.layers[i].deinit(allocator);
        }

        allocator.free(self.layers);
    }

    pub fn forward(self: *const Network, allocator: std.mem.Allocator, inputs: Matrix) !Matrix {
        var currentInput = inputs;
        var output: Matrix = undefined;

        for (0..self.numLayers) |i| {
            output = try self.layers[i].forward(allocator, currentInput);
            if (i > 0) {
                currentInput.deinit(allocator);
            }
            currentInput = output;
        }

        const final_output = try activation.softmax(allocator, output);
        output.deinit(allocator);
        return final_output;
    }
};

test "network init and deinit" {
    const allocator = std.testing.allocator;

    const numFeatures: u64 = 4;
    const config = [_]u32{ 3, 2, 1 };

    var network = try Network.init(allocator, numFeatures, &config, activation.relu);
    defer network.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 3), network.numLayers);

    try std.testing.expectEqual(@as(usize, 4), network.layers[0].numInputs); // First layer: numFeatures inputs
    try std.testing.expectEqual(@as(usize, 3), network.layers[0].numNeurons); // First layer: 3 neurons

    try std.testing.expectEqual(@as(usize, 3), network.layers[1].numInputs); // Second layer: 3 inputs (from prev layer)
    try std.testing.expectEqual(@as(usize, 2), network.layers[1].numNeurons); // Second layer: 2 neurons

    try std.testing.expectEqual(@as(usize, 2), network.layers[2].numInputs); // Third layer: 2 inputs (from prev layer)
    try std.testing.expectEqual(@as(usize, 1), network.layers[2].numNeurons); // Third layer: 1 neuron
}

test "network forward" {
    const allocator = std.testing.allocator;

    const numFeatures: u64 = 3;
    const config = [_]u32{ 3, 2 };

    var network = try Network.init(allocator, numFeatures, &config, activation.relu);
    defer network.deinit(allocator);

    // Create input
    var input = try Matrix.init(allocator, 1, 3);
    defer input.deinit(allocator);
    try input.set(0, 0, 1.0);
    try input.set(0, 1, 2.0);
    try input.set(0, 2, 3.0);

    // Set specific weights and biases for predictable output
    for (0..3) |i| {
        network.layers[0].neurons[i].weights.data[0] = 1.0;
        network.layers[0].neurons[i].weights.data[1] = 1.0;
        network.layers[0].neurons[i].weights.data[2] = 1.0;
        network.layers[0].neurons[i].bias = 0.0;
    }
    for (0..2) |i| {
        network.layers[1].neurons[i].weights.data[0] = 1.0;
        network.layers[1].neurons[i].weights.data[1] = 1.0;
        network.layers[1].neurons[i].weights.data[2] = 1.0;
        network.layers[1].neurons[i].bias = 0.0;
    }

    var result = try network.forward(allocator, input);
    defer result.deinit(allocator);

    std.debug.print("Matrix: {any}\n", .{result.data});

    // First layer: each neuron gets [1.0, 2.0, 3.0] -> outputs [6.0, 6.0, 6.0]
    // Second layer: gets [6.0, 6.0, 6.0] -> outputs [18.0, 18.0] -> [0.5, 0.5] due to softmax
    // Softmax: equal inputs should give equal probabilities
    try std.testing.expectApproxEqAbs(try result.get(0, 0), 0.5, 0.001);
    try std.testing.expectApproxEqAbs(try result.get(0, 1), 0.5, 0.001);

    // Verify softmax properties
    var sum: f64 = 0;
    for (0..result.columns) |i| {
        const value = try result.get(0, i);
        try std.testing.expect(value >= 0 and value <= 1);
        sum += value;
    }
    try std.testing.expectApproxEqAbs(sum, 1.0, 0.001);
}
