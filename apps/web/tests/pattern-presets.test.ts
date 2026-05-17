/**
 * Unit tests for the canvas's pattern preset generators.
 *
 * Every preset is a complete, runnable shape: it owns its single Start
 * node and the edge into its entry agent. Agents are keyed by name (the
 * framework bind contract), never a random id.
 */
import { describe, expect, it } from "vitest";

import { generatePattern } from "../src/lib/pattern-presets";

function startNodes(nodes: { name: string; kind?: string }[]) {
  return nodes.filter((n) => n.kind === "start");
}

describe("HUB_AND_SPOKE", () => {
  it("creates Start → hub + N-1 spokes", () => {
    const { nodes, edges, agents } = generatePattern("HUB_AND_SPOKE", 4);
    expect(nodes.map((n) => n.name)).toContain("Hub");
    // Start + Hub + 3 spokes
    expect(nodes.length).toBe(5);
    expect(startNodes(nodes).length).toBe(1);
    // Start→Hub + Hub→spoke ×3
    expect(edges.length).toBe(4);
    expect(edges).toContainEqual(
      expect.objectContaining({ source: "Start", target: "Hub" }),
    );
    for (const e of edges.filter((e) => e.source !== "Start")) {
      expect(e.source).toBe("Hub");
    }
    // agents keyed by name (Start is not an agent)
    expect(Object.keys(agents)).toContain("Hub");
    expect(Object.keys(agents)).not.toContain("Start");
  });
});

describe("PIPELINE", () => {
  it("creates Start → linear A → B → C chain", () => {
    const { nodes, edges } = generatePattern("PIPELINE", 3);
    expect(nodes.length).toBe(4); // Start + 3 agents
    expect(startNodes(nodes).length).toBe(1);
    expect(edges.length).toBe(3); // Start→A1, A1→A2, A2→A3
    expect(edges[0]).toMatchObject({ source: "Start", target: "Agent 1" });
    expect(edges[1]).toMatchObject({ source: "Agent 1", target: "Agent 2" });
    expect(edges[2]).toMatchObject({ source: "Agent 2", target: "Agent 3" });
  });
});

describe("MESH", () => {
  it("creates Start → fully-connected bidirectional mesh", () => {
    const { nodes, edges } = generatePattern("MESH", 4);
    expect(nodes.length).toBe(5); // Start + 4 agents
    expect(startNodes(nodes).length).toBe(1);
    // Start→Agent 1 (uni) + n*(n-1)/2 bidirectional pairs
    expect(edges.length).toBe(7);
    const startEdge = edges.find((e) => e.source === "Start");
    expect(startEdge).toMatchObject({ target: "Agent 1", bidirectional: false });
    for (const e of edges.filter((e) => e.source !== "Start")) {
      expect(e.bidirectional).toBe(true);
    }
  });
});

describe("HIERARCHICAL", () => {
  it("creates Start → root + managers + leaves", () => {
    const { nodes, edges } = generatePattern("HIERARCHICAL", 5);
    expect(nodes.some((n) => n.name === "Root")).toBe(true);
    expect(startNodes(nodes).length).toBe(1);
    expect(edges).toContainEqual(
      expect.objectContaining({ source: "Start", target: "Root" }),
    );
    const rootEdges = edges.filter((e) => e.source === "Root");
    expect(rootEdges.length).toBeGreaterThan(0);
    const leafEdges = edges.filter((e) => /^Leaf /.test(e.target));
    expect(leafEdges.length).toBeGreaterThan(0);
  });
});
