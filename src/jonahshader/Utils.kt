package jonahshader

public fun wrap(x: Double, bounds: Double) : Double {
    var out = x % bounds
    if (out < 0) out += bounds
    return out
}