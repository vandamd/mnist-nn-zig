const std = @import("std");
const loader = @import("loader.zig");
const Network = @import("network.zig").Network;
const activation = @import("activation.zig");
const Matrix = @import("matrix.zig").Matrix;
const loss = @import("loss.zig");

pub const Metrics = struct {
    loss: f64,
    accuracy: f64,
};

fn trainEpoch(allocator: std.mem.Allocator, network: *Network, batches: []loader.NormalisedBatch, learningRate: f64) !Metrics {
    var totalLoss: f64 = 0;
    var correctPredictions: usize = 0;
    var totalPredictions: usize = 0;

    for (batches, 0..) |batch, batchIndex| {
        var batchLoss: f64 = 0;
        var batchCorrect: usize = 0;

        for (batch.images) |image| {
            // Create input matrix from image pixels
            var inputMatrix = try Matrix.init(allocator, 1, 784);
            defer inputMatrix.deinit(allocator);
            for (0..784) |i| {
                try inputMatrix.set(0, i, image.pixels[i]);
            }

            // Create target matrix (one-hot encoding)
            var targetMatrix = try Matrix.init(allocator, 1, 10);
            defer targetMatrix.deinit(allocator);
            for (0..10) |i| {
                try targetMatrix.set(0, i, if (i == image.label) 1.0 else 0.0);
            }

            // Forward pass
            var predictions = try network.forward(allocator, inputMatrix);
            defer predictions.deinit(allocator);

            // Calculate loss
            batchLoss += try loss.crossEntropyLoss(predictions, targetMatrix);
            totalPredictions += 1;

            // Track accuracy
            var maxPred: f64 = -std.math.inf(f64);
            var predictedClass: usize = 0;
            for (0..10) |i| {
                const pred = try predictions.get(0, i);
                if (pred > maxPred) {
                    maxPred = pred;
                    predictedClass = i;
                }
            }
            if (predictedClass == image.label) {
                batchCorrect += 1;
                correctPredictions += 1;
            }

            // Backward pass and update
            var gradients = try network.backward(allocator, predictions, targetMatrix);
            defer gradients.deinit(allocator);
            try network.update(gradients, learningRate);
        }

        // Calculate and show batch metrics
        const avgBatchLoss = batchLoss / @as(f64, @floatFromInt(batch.images.len));
        const batchAccuracy = @as(f64, @floatFromInt(batchCorrect)) / @as(f64, @floatFromInt(batch.images.len));
        totalLoss += batchLoss;

        std.debug.print("\rBatch {}/{}: Loss = {d:.6}, Accuracy = {d:.2}%", .{
            batchIndex + 1,
            batches.len,
            avgBatchLoss,
            batchAccuracy * 100,
        });
    }
    std.debug.print("\n", .{}); // New line after all batches

    return Metrics{
        .loss = totalLoss / @as(f64, @floatFromInt(totalPredictions)),
        .accuracy = @as(f64, @floatFromInt(correctPredictions)) / @as(f64, @floatFromInt(totalPredictions)),
    };
}

fn validateEpoch(allocator: std.mem.Allocator, network: *Network, batches: []loader.NormalisedBatch) !Metrics {
    var totalLoss: f64 = 0;
    var correctPredictions: usize = 0;
    var totalPredictions: usize = 0;

    for (batches) |batch| {
        for (batch.images) |image| {
            // Create input matrix from image pixels
            var inputMatrix = try Matrix.init(allocator, 1, 784);
            defer inputMatrix.deinit(allocator);
            for (0..784) |i| {
                try inputMatrix.set(0, i, image.pixels[i]);
            }

            // Create target matrix (one-hot encoding)
            var targetMatrix = try Matrix.init(allocator, 1, 10);
            defer targetMatrix.deinit(allocator);
            for (0..10) |i| {
                try targetMatrix.set(0, i, if (i == image.label) 1.0 else 0.0);
            }

            // Forward pass only
            var predictions = try network.forward(allocator, inputMatrix);
            defer predictions.deinit(allocator);

            // Calculate loss
            totalLoss += try loss.crossEntropyLoss(predictions, targetMatrix);
            totalPredictions += 1;

            // Track accuracy
            var maxPred: f64 = -std.math.inf(f64);
            var predictedClass: usize = 0;
            for (0..10) |i| {
                const pred = try predictions.get(0, i);
                if (pred > maxPred) {
                    maxPred = pred;
                    predictedClass = i;
                }
            }
            if (predictedClass == image.label) {
                correctPredictions += 1;
            }
        }
    }

    return Metrics{
        .loss = totalLoss / @as(f64, @floatFromInt(totalPredictions)),
        .accuracy = @as(f64, @floatFromInt(correctPredictions)) / @as(f64, @floatFromInt(totalPredictions)),
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Training hyperparameters and configuration
    const inputSize: u64 = 784; // 28x28 pixels
    const batchSize: usize = 32;
    const numEpochs = 10;
    const learningRate = 0.001;
    const validationSplit = 0.1; // 10% for validation

    // Network architecture
    const config = [_]u32{ 128, 64, 10 }; // Hidden layers and output

    // Initialize network
    var network_instance = try Network.init(allocator, inputSize, &config, activation.ActivationType.Relu);
    defer network_instance.deinit(allocator);

    // Load and prepare data
    const images = try loader.loadTrainingData(allocator);
    defer allocator.free(images);

    // Split data into training and validation sets
    const numValidation = @as(usize, @intFromFloat(@as(f64, @floatFromInt(images.len)) * validationSplit));
    const numTraining = images.len - numValidation;

    // Create training batches
    const trainingImages = images[0..numTraining];
    loader.shuffleImages(trainingImages, null);
    const trainingBatches = try loader.createBatches(allocator, trainingImages, batchSize);
    defer loader.deinitBatches(allocator, trainingBatches);
    const normalisedTrainingBatches = try loader.createNormalisedBatches(allocator, trainingBatches);
    defer loader.deinitNormalisedBatches(allocator, normalisedTrainingBatches);

    // Create validation batches
    const validationImages = images[numTraining..];
    const validationBatches = try loader.createBatches(allocator, validationImages, batchSize);
    defer loader.deinitBatches(allocator, validationBatches);
    const normalisedValidationBatches = try loader.createNormalisedBatches(allocator, validationBatches);
    defer loader.deinitNormalisedBatches(allocator, normalisedValidationBatches);

    // Training loop
    for (0..numEpochs) |epoch| {
        std.debug.print("\n=== Epoch {}/{} ===\n", .{ epoch + 1, numEpochs });

        // Training phase
        const trainMetrics = try trainEpoch(allocator, &network_instance, normalisedTrainingBatches, learningRate);

        // Validation phase
        const validMetrics = try validateEpoch(allocator, &network_instance, normalisedValidationBatches);

        // Print detailed stats
        std.debug.print("\nTraining Results:\n", .{});
        std.debug.print("  Loss:     {d:.6}\n", .{trainMetrics.loss});
        std.debug.print("  Accuracy: {d:.2}% ({}/{} correct)\n", .{
            trainMetrics.accuracy * 100,
            @as(usize, @intFromFloat(trainMetrics.accuracy * @as(f64, @floatFromInt(numTraining)))),
            numTraining,
        });

        std.debug.print("\nValidation Results:\n", .{});
        std.debug.print("  Loss:     {d:.6}\n", .{validMetrics.loss});
        std.debug.print("  Accuracy: {d:.2}% ({}/{} correct)\n", .{
            validMetrics.accuracy * 100,
            @as(usize, @intFromFloat(validMetrics.accuracy * @as(f64, @floatFromInt(numValidation)))),
            numValidation,
        });

        std.debug.print("\nShuffling training data for next epoch...\n", .{});
        loader.shuffleImages(trainingImages, null);
    }
}
