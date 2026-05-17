/**
 * Unit tests for the canvas's pattern preset generators.
 */
import { describe, expect, it } from "vitest";

import { generatePattern } from "../src/lib/pattern-presets";

describe("HUB_AND_SPOKE", () => {
  it("creates a hub + N-1 spokes", () => {
    const { nodes, edges, agents } = generatePattern("HUB_AND_SPOKE", 4);
    expect(nodes.map((n) => n.name)).toContain("Hub");
    expect(nodes.length).toBe(4);
    expect(edges.length).toBe(3);
    for (const e of edges) {
      expect(e.source).toBe("Hub");
    }
    // Agents are keyed by name (framework bind contract), not a random id.
    expect(Object.keys(agents)).toContain("Hub");
  });
});

describe("PIPELINE", () => {
  it("creates a linear A → B → C chain", () => {
    const { nodes, edges } = generatePattern("PIPELINE", 3);
    expect(nodes.length).toBe(3);
    expect(edges.length).toBe(2);
    expect(edges[0]).toMatchObject({ source: "Agent 1", target: "Agent 2" });
    expect(edges[1]).toMatchObject({ source: "Agent 2", target: "Agent 3" });
  });
});

describe("MESH", () => {
  it("creates fully-connected bidirectional edges", () => {
    const { nodes, edges } = generatePattern("MESH", 4);
    expect(nodes.length).toBe(4);
    // n*(n-1)/2 unique pairs, each bidirectional.
    expect(edges.length).toBe(6);
    for (const e of edges) expect(e.bidirectional).toBe(true);
  });
});

describe("HIERARCHICAL", () => {
  it("creates a root + managers + leaves with edges from root to managers and managers to leaves", () => {
    const { nodes, edges } = generatePattern("HIERARCHICAL", 5);
    expect(nodes.some((n) => n.name === "Root")).toBe(true);
    const rootEdges = edges.filter((e) => e.source === "Root");
    expect(rootEdges.length).toBeGreaterThan(0);
    const leafEdges = edges.filter((e) => /^Leaf /.test(e.target));
    expect(leafEdges.length).toBeGreaterThan(0);
  });
});
