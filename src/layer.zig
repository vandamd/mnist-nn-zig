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

    pub fn init(allocator: std.mem.Allocator, numNeurons: usize, numInputs: usize, activationType: activation.ActivationType) !Layer {
        var neurons = try allocator.alloc(Neuron, numNeurons);

        for (0..numNeurons) |i| {
            neurons[i] = try Neuron.init(allocator, numInputs, activationType);
        }

        return Layer{
            .neurons = neurons,
            .numNeurons = numNeurons,
            .numInputs = numInputs,
            .activationFunction = activation.getActivationFn(activationType),
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

    pub fn backward(self: *const Layer, allocator: std.mem.Allocator, upstreamGradients: Matrix, inputs: Matrix) !LayerGradients {
        var inputGradients = try Matrix.init(allocator, 1, self.numInputs);
        var gradientsWrtWeight = try Matrix.init(allocator, self.numNeurons, self.numInputs);
        var gradientsWrtBias = try allocator.alloc(f64, self.numNeurons);
        var neuronGradients = try allocator.alloc(Neuron.Gradients, self.numNeurons);

        for (0..self.numNeurons) |i| {
            var gradient = try self.neurons[i].backward(allocator, try upstreamGradients.get(0, i), inputs);
            neuronGradients[i] = gradient;

            // Copy weight gradients into the corresponding row
            for (0..self.numInputs) |j| {
                try gradientsWrtWeight.set(i, j, try gradient.gradientWrtWeight.get(0, j));
            }
            gradientsWrtBias[i] = gradient.gradientWrtBias;

            for (0..self.numInputs) |j| {
                const currentGrad = try inputGradients.get(0, j);
                const neuronGrad = try gradient.gradientWrtInput.get(0, j);
                try inputGradients.set(0, j, currentGrad + neuronGrad);
            }
        }

        return LayerGradients{
            .gradientsWrtInput = inputGradients,
            .gradientsWrtWeight = gradientsWrtWeight,
            .gradientsWrtBias = gradientsWrtBias,
            .neuronGradients = neuronGradients,
        };
    }

    pub fn update(self: *Layer, gradients: LayerGradients, learningRate: f64) !void {
        for (0..self.numNeurons) |i| {
            try self.neurons[i].update(gradients.neuronGradients[i], learningRate);
        }
    }
};

pub const LayerGradients = struct {
    gradientsWrtInput: Matrix,
    gradientsWrtWeight: Matrix,
    gradientsWrtBias: []f64,
    neuronGradients: []Neuron.Gradients,

    pub fn deinit(self: *LayerGradients, allocator: std.mem.Allocator) void {
        self.gradientsWrtInput.deinit(allocator);
        self.gradientsWrtWeight.deinit(allocator);
        allocator.free(self.gradientsWrtBias);

        for (0..self.neuronGradients.len) |i| {
            self.neuronGradients[i].deinit(allocator);
        }
        allocator.free(self.neuronGradients);
    }
};

test "neuron layer init and deinit" {
    const allocator = std.testing.allocator;

    var layer = try Layer.init(allocator, 2, 3, activation.ActivationType.Relu);
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

    var layer = try Layer.init(allocator, 2, 3, activation.ActivationType.Relu);
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

test "layer backward" {
    const allocator = std.testing.allocator;

    var layer = try Layer.init(allocator, 2, 3, activation.ActivationType.Relu);
    defer layer.deinit(allocator);

    // Set weights and biases to known values
    for (0..layer.numNeurons) |i| {
        for (0..layer.numInputs) |j| {
            try layer.neurons[i].weights.set(0, j, 0.5); // w = 0.5 for all weights
        }
        layer.neurons[i].bias = 1.0; // b = 1.0 for all neurons
    }

    var inputs = try Matrix.init(allocator, 1, 3);
    defer inputs.deinit(allocator);
    for (0..3) |i| {
        try inputs.set(0, i, 2.0); // x = 2.0 for all inputs
    }

    // Forward pass for each neuron:
    // z = w1*x1 + w2*x2 + w3*x3 + b = 0.5*2 + 0.5*2 + 0.5*2 + 1 = 4
    // ReLU(4) = 4, ReLU'(4) = 1

    var upstreamGradients = try Matrix.init(allocator, 1, 2);
    defer upstreamGradients.deinit(allocator);
    try upstreamGradients.set(0, 0, 1.0);
    try upstreamGradients.set(0, 1, 1.0);

    var gradients = try layer.backward(allocator, upstreamGradients, inputs);
    defer gradients.deinit(allocator);

    // Each input gradient should be sum of weight*upstream for that input across neurons
    try std.testing.expectApproxEqAbs(try gradients.gradientsWrtInput.get(0, 0), 1.0, 0.001); // 0.5*1 + 0.5*1
    try std.testing.expectApproxEqAbs(try gradients.gradientsWrtInput.get(0, 1), 1.0, 0.001); // 0.5*1 + 0.5*1
    try std.testing.expectApproxEqAbs(try gradients.gradientsWrtInput.get(0, 2), 1.0, 0.001); // 0.5*1 + 0.5*1

    // Weight gradients for each neuron should be input*upstream
    for (0..layer.numInputs) |j| {
        try std.testing.expectApproxEqAbs(try gradients.gradientsWrtWeight.get(0, j), 2.0, 0.001); // neuron 1: x*1
        try std.testing.expectApproxEqAbs(try gradients.gradientsWrtWeight.get(1, j), 2.0, 0.001); // neuron 2: x*1
    }

    // Bias gradients should be 1*upstream for each neuron
    try std.testing.expectApproxEqAbs(gradients.gradientsWrtBias[0], 1.0, 0.001); // 1*1
    try std.testing.expectApproxEqAbs(gradients.gradientsWrtBias[1], 1.0, 0.001); // 1*1
}
