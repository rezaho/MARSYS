/**
 * Dagre-backed auto-layout for `@xyflow/react`.
 *
 * The brief defers Dagre vs ELK to in-flight research; we pick Dagre
 * for v0.3 (~30 KB minified vs ELK's ~400 KB; nothing >15 nodes ships
 * in v0.3). ELK becomes worth the bundle cost when topologies regularly
 * exceed 50 nodes — that's a v0.4+ decision.
 *
 * Used when a canvas loads a workflow whose nodes have no positions
 * (most commonly: a `code_import` workflow — the importer doesn't
 * compute positions). User-dragged positions persist on save.
 */
import dagre from "dagre";

const NODE_WIDTH = 220;
const NODE_HEIGHT = 92;

interface SimpleEdge {
  source: string;
  target: string;
}

interface SimpleNode {
  id: string;
  position: { x: number; y: number };
}

/**
 * Compute Dagre-laid positions for the given node + edge list.
 * Generic over the xyflow data shape via structural typing so callers
 * don't have to satisfy `Record<string, unknown>` constraints.
 */
export function autoLayout<N extends SimpleNode, E extends SimpleEdge>(
  nodes: N[],
  edges: E[],
): N[] {
  if (nodes.length === 0) return nodes;
  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({ rankdir: "LR", nodesep: 32, ranksep: 64 });

  for (const node of nodes) {
    graph.setNode(node.id, { width: NODE_WIDTH, height: NODE_HEIGHT });
  }
  for (const edge of edges) {
    graph.setEdge(edge.source, edge.target);
  }

  dagre.layout(graph);

  return nodes.map((node) => {
    const laid = graph.node(node.id);
    if (!laid) return node;
    return {
      ...node,
      position: {
        x: laid.x - NODE_WIDTH / 2,
        y: laid.y - NODE_HEIGHT / 2,
      },
    };
  });
}
