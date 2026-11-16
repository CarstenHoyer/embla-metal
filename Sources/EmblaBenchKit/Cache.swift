import EmblaMetal
import Foundation

/// `embla-bench cache <model dir> [prompt tokens] [new tokens]`: the prefix cache against computing
/// every prompt from scratch. A first request (the prompt, then `new tokens` greedy), then requests
/// that share some of it with the cache:
///
/// - turn: the next turn of a conversation (the first prompt, its reply, then more)
/// - extend: the first prompt and more (its reply not taken)
/// - fork: the first half of the first prompt, then something else
/// - miss: nothing in common
///
/// Each runs with the cache and again from scratch (`reset`) on the same engine: the tokens taken
/// from the cache, the prompt's time both ways, how many of the greedy tokens agree, and the
/// largest difference in the last step's logits. The positions taken from the cache were computed
/// by the earlier request (a decode step for its reply, the batched prefill for its prompt), so
/// the logits differ from scratch by about what the two paths differ by anyway; the tokens match.
/// Any difference in the tokens fails the check.
/// With EMBLA_BATCH=0 every prompt runs as decode steps, so both ways compute every position the
/// same way and the logits match exactly. The prompt is EMBLA_PROMPT_FILE's token ids, or a
/// sentence over and over.
func runCache(_ args: [String]) throws {
    let p = args.count > 1 ? Int(args[1])! : 1024
    let n = args.count > 2 ? Int(args[2])! : 64
    let engine = try Engine(modelDirectory: URL(fileURLWithPath: args[0]), maxContext: max(4096, 3 * p + 4 * n + 64))
    var text: [Int] = []
    if let file = ProcessInfo.processInfo.environment["EMBLA_PROMPT_FILE"] {
        text = try String(contentsOfFile: file, encoding: .utf8).split(whereSeparator: \.isWhitespace).map { Int($0)! }
    }
    while text.count < 2 * p + 2 * n { text += [758, 6426, 314, 29103, 368, 13, 198, 46447, 1307, 2201] }
    // `more` backwards, so that it is not what the model writes after `first` (its reply)
    let first = Array(text[..<p]), more = Array(text[p ..< p + n].reversed()), other = Array(text[p + n ..< 2 * p + n])

    func logits() -> [Float] {
        Array(UnsafeBufferPointer(start: engine.logits.contents().assumingMemoryBound(to: Float.self), count: Embla.vocab))
    }
    /// One request: its new tokens, the last step's logits, the prompt's seconds, tokens reused.
    func run(_ prompt: [Int]) -> ([Int], [Float], Double, Int) {
        let r = engine.generate(prompt: prompt, count: n)
        return (r.tokens, logits(), engine.prefillSeconds, engine.reusedTokens)
    }

    engine.reset()
    _ = run(first)   // warm: kernels and engram pages
    engine.reset()
    let reply = run(first).0
    let cases: [(String, [Int])] = [
        ("turn", first + reply + more),
        ("extend", first + more),
        ("fork", Array(first[..<(p / 2)]) + other),
        ("miss", other.reversed()),
    ]
    print("prompt \(p) tokens, then \(n) greedy; each request after the first one (\(p) + \(n))")
    var failed = false
    for (name, prompt) in cases {
        // the cache holds the first request's sequence before each case
        engine.reset()
        _ = run(first)
        let cached = run(prompt)
        engine.reset()
        let fresh = run(prompt)
        let agree = zip(cached.0, fresh.0).prefix { $0 == $1 }.count
        let diff = zip(cached.1, fresh.1).map { abs($0 - $1) }.max()!
        print(String(format: "  %-6@ %5d tokens, %5d from the cache · prompt %6.1f ms (from scratch %6.1f) · %d of %d tokens agree · last logits max |Δ| %.4f",
                     name as NSString, prompt.count, cached.3, cached.2 * 1e3, fresh.2 * 1e3, agree, n, diff))
        if agree < n && name != "miss" { failed = true }
        if name == "miss" && (cached.3 != 0 || diff != 0) { failed = true }
    }
    // the engine's own determinism, for scale: the same request from scratch twice
    engine.reset()
    let a = run(cases[0].1)
    engine.reset()
    let b = run(cases[0].1)
    print(String(format: "  scratch twice: last logits max |Δ| %.4f", zip(a.1, b.1).map { abs($0 - $1) }.max()!))
    print(failed ? "FAILED: a cache hit changed the tokens" : "ok")
    if failed { throw CacheCheckFailed() }
}

struct CacheCheckFailed: Error, CustomStringConvertible {
    var description: String { "the prefix cache changed the tokens" }
}
