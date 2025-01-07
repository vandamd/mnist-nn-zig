const std = @import("std");

pub const Image = struct {
    pixels: [784]u8,
    label: u8,
};

pub const NormalisedImage = struct {
    pixels: [784]f64,
    label: u8,
};

pub const Batch = struct {
    images: []Image,

    pub fn deinit(self: *Batch, allocator: std.mem.Allocator) void {
        allocator.free(self.images);
    }
};

pub const NormalisedBatch = struct {
    images: []NormalisedImage,

    pub fn deinit(self: *NormalisedBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.images);
    }
};

pub fn loadTrainingData(allocator: std.mem.Allocator) ![]Image {
    const imagesFile = try std.fs.cwd().openFile("data/train-images.idx3-ubyte", .{});
    defer imagesFile.close();
    try imagesFile.seekTo(16);
    var readerImages = imagesFile.reader();

    const labelsFile = try std.fs.cwd().openFile("data/train-labels.idx1-ubyte", .{});
    defer labelsFile.close();
    try labelsFile.seekTo(8);
    var readerLabels = labelsFile.reader();

    var images = try allocator.alloc(Image, 60000);

    for (0..60000) |i| {
        var pixels: [784]u8 = undefined;
        try readerImages.readNoEof(&pixels);

        const label = try readerLabels.readByte();

        images[i] = Image{
            .pixels = pixels,
            .label = label,
        };
    }

    return images;
}

pub fn shuffleImages(images: []Image, seed: ?u64) void {
    var prng = std.rand.DefaultPrng.init(seed orelse 1);
    const random = prng.random();

    // Fisher-Yates shuffle
    var i: usize = images.len - 1;
    while (i > 0) : (i -= 1) {
        const j = random.intRangeAtMost(usize, 0, i);
        const temp = images[i];
        images[i] = images[j];
        images[j] = temp;
    }
}

pub fn createBatches(allocator: std.mem.Allocator, images: []Image, batchSize: usize) ![]Batch {
    const numBatches = (images.len + batchSize - 1) / batchSize;
    var batches = try allocator.alloc(Batch, numBatches);

    var batchIndex: usize = 0;
    var remaining = images.len;
    var currentPos: usize = 0;

    while (remaining > 0) {
        const currentBatchSize = @min(batchSize, remaining);
        var batchImages = try allocator.alloc(Image, currentBatchSize);

        for (0..currentBatchSize) |i| {
            batchImages[i] = images[currentPos + i];
        }

        batches[batchIndex] = Batch{
            .images = batchImages,
        };

        remaining -= currentBatchSize;
        currentPos += currentBatchSize;
        batchIndex += 1;
    }

    return batches;
}

pub fn deinitBatches(allocator: std.mem.Allocator, batches: []Batch) void {
    for (batches) |batch| {
        allocator.free(batch.images);
    }
    allocator.free(batches);
}

pub fn createNormalisedBatches(allocator: std.mem.Allocator, batches: []Batch) ![]NormalisedBatch {
    var normalisedBatches = try allocator.alloc(NormalisedBatch, batches.len);

    for (batches, 0..) |batch, i| {
        var normalisedImages = try allocator.alloc(NormalisedImage, batch.images.len);

        for (batch.images, 0..) |image, j| {
            normalisedImages[j] = try normaliseImage(image);
        }

        normalisedBatches[i] = NormalisedBatch{
            .images = normalisedImages,
        };
    }

    return normalisedBatches;
}

pub fn deinitNormalisedBatches(allocator: std.mem.Allocator, batches: []NormalisedBatch) void {
    for (batches) |batch| {
        allocator.free(batch.images);
    }
    allocator.free(batches);
}

pub fn normaliseImage(image: Image) !NormalisedImage {
    var normalised: NormalisedImage = undefined;
    normalised.label = image.label;

    for (0..784) |i| {
        normalised.pixels[i] = @as(f64, @floatFromInt(image.pixels[i])) / 255.0;
    }
    return normalised;
}
