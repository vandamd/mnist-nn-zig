const std = @import("std");

pub const Image = struct {
    pixels: [784]u8,
    label: u8,
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
