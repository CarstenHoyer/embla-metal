import Foundation

/// What a model's config.json makes of `Embla`'s shape: Embla 1's constants by default, Embla 1.1's four changes
/// (docs/EMBLA_1_1.md) when its keys are there. The layer kinds, widths and heads stay Embla 1's.
public struct Arch: Sendable {
    /// Full layer -> the layer whose (normed, mixed) input its K/V project (`kv_sources`).
    public var kvSources: [Int: Int] = Embla.kvSources
    /// K/V source layers whose own K/V project a mix of their own (`kv_mix_layers`): its column right after
    /// the layer's attention query, its row 1 of the layer's mixes.
    public var kvMix: Set<Int> = []
    /// Layers [0, routerMix) whose MoE router scores a mix of its own (`moe_router_mix`, `moe_router_mix_layers`).
    public var routerMix = 0
    /// Layers with a dense MLP (`mlp_layer_types`): Embla 1's layer 0.
    public var dense: Set<Int> = [0]
    /// Each layer's shared expert width (`shared_expert_intermediate_size`, `shared_expert_layer_sizes`).
    public var shared = [Int](repeating: Embla.sharedDim, count: Embla.layers)
    /// The looped layers (`loop_layers`, two passes, full form): Embla 1.1's [8, 16).
    public var loop: Range<Int>? = nil

    /// K/V source layer -> the later full layers that project its input.
    public var bridges: [Int: [Int]] {
        var b: [Int: [Int]] = [:]
        for (f, s) in kvSources.sorted(by: { $0.key < $1.key }) where f != s { b[s, default: []].append(f) }
        return b
    }
    /// Source slots a token: emb, engram, 7 blocks (Embla 1); the loop adds b* of each looped block and, while
    /// the second pass runs, its first block (slot 6 before b* takes it).
    public var slots: Int { loop == nil ? 9 : 12 }
    /// Looped blocks.
    public var loopBlocks: Int { loop.map { $0.count / Embla.blockSize } ?? 0 }
    /// Sources before layer i's first sub-layer (pass 1), its block's completed ones included.
    public func sources(_ i: Int) -> Int {
        i == 0 ? 1 : 2 + i / Embla.blockSize + (loop.map { i >= $0.upperBound ? loopBlocks : 0 } ?? 0)
    }
    /// The K mixes of layer i's attention: its query, its own K/V mix, its bridged layers'.
    public func attnK(_ i: Int) -> Int { 1 + (kvMix.contains(i) ? 1 : 0) + (bridges[i]?.count ?? 0) }
    public func routerK(_ i: Int) -> Int { !dense.contains(i) && i < routerMix ? 2 : 1 }
    /// Mixes a token's mix output holds (the most any layer's attention takes, at least Embla 1's 3).
    public var outMixes: Int { max(3, (0 ..< Embla.layers).map(attnK).max()!) }

    /// AttnRes pseudo-query columns (Engine.buildPlans' order).
    public var queryCount: Int {
        (0 ..< Embla.layers).reduce(2) { $0 + attnK($1) + routerK($1) } + (loop ?? 0 ..< 0).reduce(0) { $0 + 1 + routerK($1) }
    }

    public static let embla1 = Arch()

    /// From a model directory's config.json; Embla 1 without one.
    public static func load(_ dir: URL) throws -> Arch {
        let url = dir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url) else { return .embla1 }
        guard let c = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw EmblaError.badFile(url.path) }
        var a = Arch()
        func int(_ v: Any?) -> Int? { (v as? NSNumber)?.intValue ?? (v as? String).flatMap { Int($0) } }
        if let kv = c["kv_sources"] as? [String: Any] {
            a.kvSources = Dictionary(uniqueKeysWithValues: kv.compactMap { k, v in Int(k).flatMap { kk in int(v).map { (kk, $0) } } })
        }
        a.kvMix = Set((c["kv_mix_layers"] as? [Any] ?? []).compactMap(int))
        if (c["moe_router_mix"] as? Bool) == true { a.routerMix = int(c["moe_router_mix_layers"]) ?? Embla.layers }
        if let t = c["mlp_layer_types"] as? [String] {
            a.dense = Set(t.enumerated().filter { $0.element == "dense" }.map(\.offset))
        }
        if let w = int(c["shared_expert_intermediate_size"]) { a.shared = [Int](repeating: w, count: Embla.layers) }
        for (k, v) in c["shared_expert_layer_sizes"] as? [String: Any] ?? [:] {
            if let i = Int(k), let w = int(v) { a.shared[i] = w }
        }
        if let l = c["loop_layers"] as? [Any], l.count == 2, let lo = int(l[0]), let hi = int(l[1]) {
            a.loop = lo ..< hi
            let passes = int(c["loop_passes"]) ?? 2, form = c["loop_pass_form"] as? String ?? "full"
            guard passes == 2, form == "full" else { throw EmblaError.unsupported("loop: \(passes) passes, form \(form)") }
        }
        try a.validate()
        return a
    }

    /// What this engine's kernels and plans take.
    func validate() throws {
        func no(_ why: String) throws { throw EmblaError.unsupported(why) }
        for (f, s) in kvSources where Embla.kinds[f] != .full || Embla.kinds[s] != .full || kvSources[s] != s || s > f { try no("kv_sources \(f): \(s)") }
        if Set(kvSources.keys) != Set((0 ..< Embla.layers).filter { Embla.kinds[$0] == .full }) { try no("kv_sources: every full layer") }
        for i in kvMix where kvSources[i] != i { try no("kv_mix_layers: \(i) is not a K/V source") }
        if kvSources.values.contains(where: { $0 > Embla.prefillLayers }) { try no("a K/V source past the prefill half") }
        if (0 ..< Embla.layers).contains(where: { attnK($0) > 5 }) { try no("more than 5 attention mixes a layer") }
        for i in dense where i != 0 { try no("dense MLP at layer \(i)") }
        for (i, w) in shared.enumerated() where (w != 1024 && w != 2048) || (w == 2048 && i != 0) { try no("shared expert width \(w) at layer \(i)") }
        if let l = loop {
            // two whole AttnRes blocks of sliding layers in the prefill half (their second pass needs no full K/V)
            if l != 8 ..< 16 { try no("loop_layers \(l): only [8, 16)") }
            if l.contains(where: { Embla.kinds[$0] != .sliding }) { try no("loop over a full layer") }
        }
    }
}
