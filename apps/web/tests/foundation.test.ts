import { describe, expect, it } from "vitest";
import type { BootstrapResponse } from "../src/lib/api";

describe("foundation smoke", () => {
  it("BootstrapResponse type shape compiles", () => {
    const sample: BootstrapResponse = {
      framework: { version: "0.2.1-beta" },
      spren: { active: false, version: "0.3.0" },
      surfaces: ["gui"],
      capabilities: {},
      endpoints: {},
      started_at: new Date().toISOString(),
      data_dir: "/tmp/spren",
    };
    expect(sample.spren.version).toBe("0.3.0");
    expect(sample.surfaces).toContain("gui");
  });
});
