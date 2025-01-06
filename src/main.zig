const std = @import("std");
const vis = @import("visualiser.zig");
const loader = @import("loader.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const images = try loader.loadTrainingData(allocator);
    defer allocator.free(images);

    try vis.displayImage(allocator, images[0], null);
    std.debug.print("\nLabel: {}\n", .{images[0].label});
}
