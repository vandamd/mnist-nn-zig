const std = @import("std");

pub const ActivationFn = *const fn (f64) f64;

pub fn relu(x: f64) f64 {
    return @max(0, x);
}

pub fn reluDerivative(x: f64) f64 {
    if (x > 0) {
        return 1;
    } else {
        return 0;
    }
}

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + std.math.exp(-x));
}

pub fn sigmoidDerivative(x: f64) f64 {
    return sigmoid(x) * (1 - sigmoid(x));
}
