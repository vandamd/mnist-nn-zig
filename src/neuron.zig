const std = @import("std");

const Matrix = @import("matrix.zig").Matrix;
const operations = @import("operations.zig");
const activation = @import("activation.zig");

pub const Neuron = struct {
    weights: Matrix,
    bias: f64,
    activationType: activation.ActivationType,
    activationFunction: activation.ActivationFn,
    activationFunctionDeriv: activation.ActivationFn,
    prng: std.rand.DefaultPrng,

    pub const Gradients = struct {
        gradientWrtInput: Matrix,
        gradientWrtWeight: Matrix,
        gradientWrtBias: f64,

        pub fn deinit(self: *Gradients, allocator: std.mem.Allocator) void {
            self.gradientWrtInput.deinit(allocator);
            self.gradientWrtWeight.deinit(allocator);
        }
    };

    pub fn init(allocator: std.mem.Allocator, numInputs: usize, activationType: activation.ActivationType) !Neuron {
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
            .activationType = activationType,
            .activationFunction = activation.getActivationFn(activationType),
            .activationFunctionDeriv = activation.getActivationDerivFn(activationType),
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
        const value = self.activationFunction(sum + self.bias);
        return value;
    }

    pub fn backward(self: *const Neuron, allocator: std.mem.Allocator, upstreamGradient: f64, inputs: Matrix) !Gradients {
        var inputsTransposed = try operations.transpose(allocator, inputs);
        defer inputsTransposed.deinit(allocator);

        // The loss depends on three things:
        // 1. The weight of the connection
        // 2. The input strength/amount
        // 3. The bias used

        // We use chain rule to find three derivatives of the loss wrt to those
        // three things!
        // 1. dC/dw = dz/dw * da(l)/dz * dC/da(l)
        // 1. dC/db = dz/db * da(l)/dz * dC/da(l)
        // 1. dC/da(l-1) = dz/da(l-1) * da(l)/dz * dC/da(l)

        var mult = try operations.multiply(allocator, self.weights, inputsTransposed);
        defer mult.deinit(allocator);

        const sum: f64 = try mult.get(0, 0);
        const z = sum + self.bias;

        const activationDeriv = self.activationFunctionDeriv(z); // da(l)/dz or σ'(z)

        var slopeWrtInput = try Matrix.init(allocator, 1, inputs.columns);
        var slopeWrtWeight = try Matrix.init(allocator, 1, inputs.columns);

        for (0..slopeWrtInput.columns) |i| {
            const sumDerivWrtInput = try self.weights.get(0, i);
            try slopeWrtInput.set(0, i, sumDerivWrtInput * activationDeriv * upstreamGradient);

            const sumDerivWrtWeight = try inputs.get(0, i);
            try slopeWrtWeight.set(0, i, sumDerivWrtWeight * activationDeriv * upstreamGradient);
        }

        const biasGradient = activationDeriv * upstreamGradient;

        return Gradients{
            .gradientWrtInput = slopeWrtInput,
            .gradientWrtWeight = slopeWrtWeight,
            .gradientWrtBias = biasGradient,
        };
    }

    pub fn update(self: *Neuron, gradients: Gradients, learningRate: f64) !void {
        // Update weights: w = w - learning_rate * dw
        for (0..self.weights.columns) |i| {
            const weightGrad = try gradients.gradientWrtWeight.get(0, i);
            const currentWeight = try self.weights.get(0, i);
            try self.weights.set(0, i, currentWeight - learningRate * weightGrad);
        }

        // Update bias: b = b - learning_rate * db
        self.bias = self.bias - learningRate * gradients.gradientWrtBias;
    }
};

test "neuron init and deinit" {
    const allocator = std.testing.allocator;

    var neuron = try Neuron.init(allocator, 3, activation.ActivationType.Relu);
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

    var neuron = try Neuron.init(allocator, 3, activation.ActivationType.Relu);
    defer neuron.deinit(allocator);

    var input = try Matrix.init(allocator, 1, 3);
    defer input.deinit(allocator);

    for (0..3) |i| {
        try neuron.weights.set(0, i, 0.5);
        try input.set(0, i, 2.0);
    }

    neuron.bias = 1.0;

    const result = try neuron.forward(allocator, input);
    try std.testing.expectApproxEqAbs(result, 4.0, 0.001);
}

test "neuron backward" {
    const allocator = std.testing.allocator;

    var neuron = try Neuron.init(allocator, 2, activation.ActivationType.Relu);
    defer neuron.deinit(allocator);

    try neuron.weights.set(0, 0, 0.5); // w1 = 0.5
    try neuron.weights.set(0, 1, -0.5); // w2 = -0.5
    neuron.bias = 1.0;

    var inputs = try Matrix.init(allocator, 1, 2);
    defer inputs.deinit(allocator);
    try inputs.set(0, 0, 2.0); // x1 = 2.0
    try inputs.set(0, 1, 3.0); // x2 = 3.0

    // Forward pass: z = w1*x1 + w2*x2 + b = 0.5*2 + (-0.5*3) + 1 = 0.5
    // ReLU(0.5) = 0.5, ReLU'(0.5) = 1

    var gradients = try neuron.backward(allocator, 1.0, inputs);
    defer gradients.gradientWrtInput.deinit(allocator);
    defer gradients.gradientWrtWeight.deinit(allocator);

    try std.testing.expectApproxEqAbs(try gradients.gradientWrtInput.get(0, 0), 0.5, 0.001); // w1 * 1 * upstream
    try std.testing.expectApproxEqAbs(try gradients.gradientWrtInput.get(0, 1), -0.5, 0.001); // w2 * 1 * upstream
    try std.testing.expectApproxEqAbs(try gradients.gradientWrtWeight.get(0, 0), 2.0, 0.001); // x1 * 1 * upstream
    try std.testing.expectApproxEqAbs(try gradients.gradientWrtWeight.get(0, 1), 3.0, 0.001); // x2 * 1 * upstream
    try std.testing.expectApproxEqAbs(gradients.gradientWrtBias, 1.0, 0.001); // 1 * 1 * upstream
}

test "neuron update" {
    const allocator = std.testing.allocator;

    var neuron = try Neuron.init(allocator, 2, activation.ActivationType.Relu);
    defer neuron.deinit(allocator);

    try neuron.weights.set(0, 0, 0.5);
    try neuron.weights.set(0, 1, -0.5);
    neuron.bias = 1.0;

    var gradientWrtWeight = try Matrix.init(allocator, 1, 2);
    defer gradientWrtWeight.deinit(allocator);
    try gradientWrtWeight.set(0, 0, 2.0);
    try gradientWrtWeight.set(0, 1, 3.0);

    const gradients = Neuron.Gradients{
        .gradientWrtInput = undefined, // Not needed for update
        .gradientWrtWeight = gradientWrtWeight,
        .gradientWrtBias = 1.0,
    };

    const learningRate = 0.1;
    try neuron.update(gradients, learningRate);

    // Check updated weights: w = w - lr * gradient
    try std.testing.expectApproxEqAbs(try neuron.weights.get(0, 0), 0.3, 0.001); // 0.5 - 0.1 * 2.0
    try std.testing.expectApproxEqAbs(try neuron.weights.get(0, 1), -0.8, 0.001); // -0.5 - 0.1 * 3.0
    try std.testing.expectApproxEqAbs(neuron.bias, 0.9, 0.001); // 1.0 - 0.1 * 1.0
}
