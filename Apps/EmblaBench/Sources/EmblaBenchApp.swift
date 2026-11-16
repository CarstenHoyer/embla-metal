import EmblaBenchKit
import Foundation
import SwiftUI

/// The smallest iPhone host for `embla-bench`: launched with `bench <command> <args...>` (devicectl
/// `process launch --console`), it runs the command (`runBench`) and exits; its output is the console.
/// An argument `docs:<path>` is a path in the app's Documents (where `devicectl device copy to` puts
/// files), as is a relative EMBLA_PROMPT_FILE. `link <target> <link>` makes a symlink in Documents (a
/// model directory sharing another's engram table). Every run logs Low Power Mode and the thermal
/// state before and after.
@main
struct EmblaBenchApp: App {
    init() {
        let args = Array(ProcessInfo.processInfo.arguments.dropFirst())
        guard let verb = args.first, verb == "bench" || verb == "link" else { return }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let path: @Sendable (String) -> String = { a in
            a.hasPrefix("docs:") ? docs.appendingPathComponent(String(a.dropFirst(5))).path : a
        }
        setvbuf(stdout, nil, _IOLBF, 0)
        Thread.detachNewThread {
            if verb == "link" {
                do {
                    try? FileManager.default.removeItem(atPath: path(args[2]))
                    try FileManager.default.createSymbolicLink(atPath: path(args[2]), withDestinationPath: path(args[1]))
                    print("linked \(path(args[2])) -> \(path(args[1]))")
                } catch { print("ERROR \(error)") }
                exit(0)
            }
            if let file = ProcessInfo.processInfo.environment["EMBLA_PROMPT_FILE"], !file.hasPrefix("/") {
                setenv("EMBLA_PROMPT_FILE", docs.appendingPathComponent(file).path, 1)
            }
            @Sendable func device() -> String {
                let p = ProcessInfo.processInfo
                let thermal = ["nominal", "fair", "serious", "critical"][p.thermalState.rawValue]
                return "low_power=\(p.isLowPowerModeEnabled ? 1 : 0) thermal=\(thermal)"
            }
            print("EMBLA_BENCH start \(device())")
            do { try runBench(args.dropFirst().map(path)) } catch { print("ERROR \(error)") }
            print("EMBLA_BENCH done \(device())")
            exit(0)
        }
    }

    var body: some Scene {
        WindowGroup { Text("EmblaBench: launch with `bench <embla-bench command> ...`") }
    }
}
