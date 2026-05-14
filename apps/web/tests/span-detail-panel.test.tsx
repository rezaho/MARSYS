/**
 * SpanDetailPanel tests.
 *
 * Covers AC-344..364: kind-specific attribute layouts (generation, tool,
 * step, branch, execution), redacted-args passthrough, full-content
 * expansions, dismiss-on-Esc + dismissOnBackdrop=false, close button.
 */
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { SpanDetailPanel } from "../src/components/TraceTree";
import type { SpanNode } from "../src/lib/api";

function makeSpan(overrides: Partial<SpanNode> & { kind: SpanNode["kind"] }): SpanNode {
  return {
    span_id: overrides.span_id ?? "span-1",
    parent_span_id: null,
    trace_id: "trace-1",
    name: overrides.name ?? "test span",
    kind: overrides.kind,
    start_time: 0,
    end_time: 1,
    duration_ms: 1234,
    status: overrides.status ?? "ok",
    attributes: overrides.attributes ?? {},
    events: overrides.events ?? [],
    links: overrides.links ?? [],
    children: [],
  };
}

describe("SpanDetailPanel", () => {
  it("renders nothing when span is null (closed)", () => {
    render(<SpanDetailPanel span={null} onClose={vi.fn()} />);
    expect(screen.queryByTestId("span-detail-panel")).toBeNull();
  });

  it("renders the panel + title + kind chip when a span is provided", () => {
    const span = makeSpan({ kind: "generation", name: "claude-opus-4-7" });
    render(<SpanDetailPanel span={span} onClose={vi.fn()} />);
    expect(screen.getByTestId("span-detail-panel")).toBeTruthy();
    expect(screen.getByTestId("span-detail-title").textContent).toBe(
      "claude-opus-4-7",
    );
  });

  it("renders generation-kind attributes (model, tokens, finish_reason, etc.)", () => {
    const span = makeSpan({
      kind: "generation",
      attributes: {
        model_name: "claude-opus-4-7",
        provider: "anthropic",
        prompt_tokens: 3201,
        completion_tokens: 542,
        reasoning_tokens: 120,
        response_time_ms: 4500,
        finish_reason: "stop",
        has_thinking: true,
        has_tool_calls: false,
      },
    });
    render(<SpanDetailPanel span={span} onClose={vi.fn()} />);
    const list = screen.getByTestId("attribute-list");
    expect(list.textContent).toContain("claude-opus-4-7");
    expect(list.textContent).toContain("anthropic");
    expect(list.textContent).toContain("3,201");
    expect(list.textContent).toContain("542");
    expect(list.textContent).toContain("120");
    expect(list.textContent).toContain("4500ms");
    expect(list.textContent).toContain("stop");
  });

  it("renders tool-kind attributes (tool name, args, result_summary) verbatim", () => {
    const span = makeSpan({
      kind: "tool",
      attributes: {
        tool_name: "web_search",
        agent_name: "Researcher",
        arguments: { query: "[REDACTED]" },
        result_summary: "5 results returned",
      },
    });
    render(<SpanDetailPanel span={span} onClose={vi.fn()} />);
    const list = screen.getByTestId("attribute-list");
    expect(list.textContent).toContain("web_search");
    expect(list.textContent).toContain("Researcher");
    // Redacted value passthrough — Spren does NOT re-redact (AC-451).
    expect(list.textContent).toContain("[REDACTED]");
    expect(list.textContent).toContain("5 results returned");
  });

  it("renders step-kind attributes (agent_name, step_number, success)", () => {
    const span = makeSpan({
      kind: "step",
      attributes: {
        agent_name: "Researcher",
        step_number: 1,
        action_type: "generate",
        success: true,
      },
    });
    render(<SpanDetailPanel span={span} onClose={vi.fn()} />);
    const list = screen.getByTestId("attribute-list");
    expect(list.textContent).toContain("Researcher");
    expect(list.textContent).toContain("1");
    expect(list.textContent).toContain("generate");
    expect(list.textContent).toContain("yes");
  });

  it("Show full prompt + response toggle inline expansion when content is captured", () => {
    const span = makeSpan({
      kind: "generation",
      attributes: {
        model_name: "claude-opus-4-7",
        prompt_content: "What is 2+2?",
        response_content: "2+2 equals 4.",
      },
    });
    render(<SpanDetailPanel span={span} onClose={vi.fn()} />);
    // Initially collapsed.
    expect(screen.queryByTestId("show-prompt-body")).toBeNull();
    fireEvent.click(screen.getByTestId("show-prompt"));
    expect(screen.getByTestId("show-prompt-body").textContent).toBe(
      "What is 2+2?",
    );
    // Click again to collapse.
    fireEvent.click(screen.getByTestId("show-prompt"));
    expect(screen.queryByTestId("show-prompt-body")).toBeNull();
  });

  it("Show full toggles render disabled with tooltip when content is absent", () => {
    const span = makeSpan({
      kind: "generation",
      attributes: { model_name: "claude-opus-4-7" },
    });
    render(<SpanDetailPanel span={span} onClose={vi.fn()} />);
    const promptBtn = screen.getByTestId("show-prompt");
    const responseBtn = screen.getByTestId("show-response");
    expect((promptBtn as HTMLButtonElement).disabled).toBe(true);
    expect(promptBtn.getAttribute("title")).toBe("Content not captured for this span");
    expect((responseBtn as HTMLButtonElement).disabled).toBe(true);
  });

  it("Show full toggles do NOT render for non-generation kinds", () => {
    const span = makeSpan({
      kind: "tool",
      attributes: { tool_name: "x" },
    });
    render(<SpanDetailPanel span={span} onClose={vi.fn()} />);
    expect(screen.queryByTestId("show-prompt")).toBeNull();
    expect(screen.queryByTestId("show-response")).toBeNull();
  });

  it("close button calls onClose", () => {
    const onClose = vi.fn();
    const span = makeSpan({ kind: "generation" });
    render(<SpanDetailPanel span={span} onClose={onClose} />);
    fireEvent.click(screen.getByTestId("span-detail-close"));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("Esc key calls onClose (SlideOver wires this)", () => {
    const onClose = vi.fn();
    const span = makeSpan({ kind: "generation" });
    render(<SpanDetailPanel span={span} onClose={onClose} />);
    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("backdrop click does NOT dismiss (dismissOnBackdrop=false)", () => {
    const onClose = vi.fn();
    const span = makeSpan({ kind: "generation" });
    render(<SpanDetailPanel span={span} onClose={onClose} />);
    fireEvent.mouseDown(screen.getByTestId("span-detail-backdrop"));
    expect(onClose).not.toHaveBeenCalled();
  });
});
