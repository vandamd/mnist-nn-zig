const std = @import("std");

const layer = @import("layer.zig");
const Layer = @import("layer.zig").Layer;
const Neuron = @import("neuron.zig").Neuron;
const Matrix = @import("matrix.zig").Matrix;
const operations = @import("operations.zig");
const activation = @import("activation.zig");
const loss = @import("loss.zig");

pub const NetworkError = error{
    tooFewLayers,
};

pub const Network = struct {
    layers: []Layer,
    numLayers: usize,
    neuronConfig: []const u32,
    layerInputs: ?[]Matrix,

    pub fn init(allocator: std.mem.Allocator, numFeatures: u64, neuronConfig: []const u32, activationType: activation.ActivationType) !Network {
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

            layers[i] = try Layer.init(allocator, neuronConfig[i], numInputs, activationType);
        }

        return Network{
            .layers = layers,
            .numLayers = numLayers,
            .neuronConfig = neuronConfig,
            .layerInputs = null,
        };
    }

    pub fn deinit(self: *Network, allocator: std.mem.Allocator) void {
        for (0..self.numLayers) |i| {
            self.layers[i].deinit(allocator);
            if (self.layerInputs != null) {
                self.layerInputs.?[i].deinit(allocator);
            }
        }

        if (self.layerInputs != null) {
            allocator.free(self.layerInputs.?);
        }
        allocator.free(self.layers);
    }

    pub fn forward(self: *Network, allocator: std.mem.Allocator, inputs: Matrix) !Matrix {
        if (self.layerInputs == null) {
            self.layerInputs = try allocator.alloc(Matrix, self.numLayers);
        }

        var currentInput = inputs;
        var output: Matrix = undefined;

        for (0..self.numLayers) |i| {
            self.layerInputs.?[i] = try currentInput.copy(allocator);

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

    pub fn backward(self: *const Network, allocator: std.mem.Allocator, predictions: Matrix, targets: Matrix) !NetworkGradients {
        var layerGradients = try allocator.alloc(layer.LayerGradients, self.numLayers);

        // Get initial gradient from cross entropy loss
        var currentGradient = try loss.crossEntropyGradient(allocator, predictions, targets);
        defer currentGradient.deinit(allocator);

        // Iterate through layers in reverse order
        var i: usize = self.numLayers;
        while (i > 0) {
            i -= 1;

            // Get the inputs that were used for this layer
            const layerInputs = self.layerInputs.?[i];

            // Compute gradients for this layer
            layerGradients[i] = try self.layers[i].backward(allocator, currentGradient, layerInputs);

            // The gradient with respect to inputs becomes the upstream gradient for the next (previous) layer
            if (i > 0) {
                const nextGradient = try layerGradients[i].gradientsWrtInput.copy(allocator);
                currentGradient.deinit(allocator);
                currentGradient = nextGradient;
            }
        }

        return NetworkGradients{
            .layerGradients = layerGradients,
        };
    }

    pub fn update(self: *Network, gradients: NetworkGradients, learningRate: f64) !void {
        for (0..self.numLayers) |i| {
            try self.layers[i].update(gradients.layerGradients[i], learningRate);
        }
    }
};

pub const NetworkGradients = struct {
    layerGradients: []layer.LayerGradients,

    pub fn deinit(self: *NetworkGradients, allocator: std.mem.Allocator) void {
        for (self.layerGradients) |*gradient| {
            gradient.deinit(allocator);
        }
        allocator.free(self.layerGradients);
    }
};

test "network init and deinit" {
    const allocator = std.testing.allocator;

    const numFeatures: u64 = 4;
    const config = [_]u32{ 3, 2, 1 };

    var network = try Network.init(allocator, numFeatures, &config, activation.ActivationType.Relu);
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

    var network = try Network.init(allocator, numFeatures, &config, activation.ActivationType.Relu);
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

    // Verify layerInputs were stored correctly
    try std.testing.expect(network.layerInputs != null);
    try std.testing.expectApproxEqAbs(try network.layerInputs.?[0].get(0, 0), 1.0, 0.001);
    try std.testing.expectApproxEqAbs(try network.layerInputs.?[0].get(0, 1), 2.0, 0.001);
    try std.testing.expectApproxEqAbs(try network.layerInputs.?[0].get(0, 2), 3.0, 0.001);

    try std.testing.expectApproxEqAbs(try network.layerInputs.?[1].get(0, 0), 6.0, 0.001);
    try std.testing.expectApproxEqAbs(try network.layerInputs.?[1].get(0, 1), 6.0, 0.001);
    try std.testing.expectApproxEqAbs(try network.layerInputs.?[1].get(0, 2), 6.0, 0.001);

    // Verify softmax properties
    var sum: f64 = 0;
    for (0..result.columns) |i| {
        const value = try result.get(0, i);
        try std.testing.expect(value >= 0 and value <= 1);
        sum += value;
    }
    try std.testing.expectApproxEqAbs(sum, 1.0, 0.001);
}

test "network backward" {
    const allocator = std.testing.allocator;

    const numFeatures: u64 = 3;
    const config = [_]u32{ 3, 2 };

    var network = try Network.init(allocator, numFeatures, &config, activation.ActivationType.Relu);
    defer network.deinit(allocator);

    // Create input
    var input = try Matrix.init(allocator, 1, 3);
    defer input.deinit(allocator);
    try input.set(0, 0, 1.0);
    try input.set(0, 1, 2.0);
    try input.set(0, 2, 3.0);

    // Set weights and biases to known values
    for (0..3) |i| {
        network.layers[0].neurons[i].weights.data[0] = 0.5;
        network.layers[0].neurons[i].weights.data[1] = 0.5;
        network.layers[0].neurons[i].weights.data[2] = 0.5;
        network.layers[0].neurons[i].bias = 1.0;
    }
    for (0..2) |i| {
        network.layers[1].neurons[i].weights.data[0] = 0.5;
        network.layers[1].neurons[i].weights.data[1] = 0.5;
        network.layers[1].neurons[i].weights.data[2] = 0.5;
        network.layers[1].neurons[i].bias = 1.0;
    }

    // Forward pass
    var predictions = try network.forward(allocator, input);
    defer predictions.deinit(allocator);

    // Create target values
    var targets = try Matrix.init(allocator, 1, 2);
    defer targets.deinit(allocator);
    try targets.set(0, 0, 1.0);
    try targets.set(0, 1, 0.0);

    // Backward pass
    var gradients = try network.backward(allocator, predictions, targets);
    defer gradients.deinit(allocator);

    // Verify gradients exist for each layer
    try std.testing.expectEqual(network.numLayers, gradients.layerGradients.len);

    // Verify gradient shapes
    try std.testing.expectEqual(@as(usize, 1), gradients.layerGradients[0].gradientsWrtInput.rows);
    try std.testing.expectEqual(@as(usize, 3), gradients.layerGradients[0].gradientsWrtInput.columns);
    try std.testing.expectEqual(@as(usize, 3), gradients.layerGradients[0].gradientsWrtWeight.rows);
    try std.testing.expectEqual(@as(usize, 3), gradients.layerGradients[0].gradientsWrtWeight.columns);
    try std.testing.expectEqual(@as(usize, 3), gradients.layerGradients[0].gradientsWrtBias.len);

    try std.testing.expectEqual(@as(usize, 1), gradients.layerGradients[1].gradientsWrtInput.rows);
    try std.testing.expectEqual(@as(usize, 3), gradients.layerGradients[1].gradientsWrtInput.columns);
    try std.testing.expectEqual(@as(usize, 2), gradients.layerGradients[1].gradientsWrtWeight.rows);
    try std.testing.expectEqual(@as(usize, 3), gradients.layerGradients[1].gradientsWrtWeight.columns);
    try std.testing.expectEqual(@as(usize, 2), gradients.layerGradients[1].gradientsWrtBias.len);

    // Verify gradients are non-zero
    var hasNonZeroGradient = false;
    for (0..gradients.layerGradients[1].gradientsWrtWeight.rows) |i| {
        for (0..gradients.layerGradients[1].gradientsWrtWeight.columns) |j| {
            const grad = try gradients.layerGradients[1].gradientsWrtWeight.get(i, j);
            if (grad != 0) {
                hasNonZeroGradient = true;
                break;
            }
        }
    }
    try std.testing.expect(hasNonZeroGradient);
}
