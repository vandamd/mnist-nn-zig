const std = @import("std");
const Image = @import("loader.zig").Image;

pub fn displayImage(allocator: std.mem.Allocator, image: Image, scale: ?u32) !void {
    const stdout = std.io.getStdOut().writer();
    const s = scale orelse 20;
    const imageSize = 28 * s;

    var rgbData = try allocator.alloc(u8, imageSize * imageSize * 3);
    defer allocator.free(rgbData);

    // Convert to RGB
    for (0..imageSize * imageSize) |i| {
        const sourcePixelX = i % imageSize / s;
        const sourcePixelY = i / imageSize / s;
        const pixel = image.pixels[sourcePixelY * 28 + sourcePixelX];

        rgbData[i * 3] = pixel; // R
        rgbData[i * 3 + 1] = pixel; // G
        rgbData[i * 3 + 2] = pixel; // B
    }

    // Kitty image command
    try stdout.writeAll("\x1b_G");
    try stdout.print("f=24,a=T,s={},v={};", .{ imageSize, imageSize });

    const encoded_buf = try allocator.alloc(u8, std.base64.standard.Encoder.calcSize(imageSize * imageSize * 3));
    defer allocator.free(encoded_buf);
    const encoded = std.base64.standard.Encoder.encode(encoded_buf, rgbData);
    try stdout.writeAll(encoded);

    // End Kitty command
    try stdout.writeAll("\x1b\\");
}
