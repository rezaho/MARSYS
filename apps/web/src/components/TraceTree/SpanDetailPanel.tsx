/**
 * Right-drawer span detail panel.
 *
 * Built on the ``SlideOver`` primitive with ``side="right"``,
 * ``width={320}``, and ``dismissOnBackdrop={false}`` — accidental
 * backdrop clicks while reading attributes do NOT dismiss the
 * drawer. Esc dismisses (handled by SlideOver's ``useEscapeKey``).
 *
 * Kind-specific attribute layouts per
 * ``docs/architecture/spren/06-observability.md`` §Attributes by
 * kind. Long fields collapse under "Show full" expand buttons.
 */
import { useState, type ReactElement } from "react";

import type { SpanNode } from "../../lib/api";
import { SlideOver } from "../ui/SlideOver";

import "./SpanDetailPanel.css";

export interface SpanDetailPanelProps {
  span: SpanNode | null;
  onClose: () => void;
}

export function SpanDetailPanel({ span, onClose }: SpanDetailPanelProps): ReactElement {
  return (
    <SlideOver
      open={span !== null}
      onClose={onClose}
      ariaLabel="Span details"
      side="right"
      width={320}
      className="span-detail-panel"
      testId="span-detail-panel"
      backdropTestId="span-detail-backdrop"
      dismissOnBackdrop={false}
    >
      {span ? <SpanDetailBody span={span} onClose={onClose} /> : <></>}
    </SlideOver>
  );
}

function SpanDetailBody({
  span,
  onClose,
}: {
  span: SpanNode;
  onClose: () => void;
}): ReactElement {
  return (
    <div className="span-detail-body">
      <header className="span-detail-header">
        <span className="span-detail-kind" data-kind={span.kind}>
          {span.kind}
        </span>
        <button
          type="button"
          className="span-detail-close"
          onClick={onClose}
          aria-label="Close panel"
          data-testid="span-detail-close"
        >
          ✕
        </button>
      </header>
      <h2 className="span-detail-title" data-testid="span-detail-title">
        {span.name}
      </h2>
      <div className="span-detail-divider" />

      <AttributeList kind={span.kind} attributes={span.attributes ?? {}} duration_ms={span.duration_ms} />

      <ContentExpansions kind={span.kind} attributes={span.attributes ?? {}} />

      <Section
        label="Events"
        empty={(span.events ?? []).length === 0}
        emptyText="(none)"
        data-testid="span-detail-events"
      >
        {(span.events ?? []).map((evt, idx) => (
          <div key={idx} className="span-detail-event">
            <span className="span-detail-event-name">{(evt as { name?: string })?.name ?? "event"}</span>
            <pre className="span-detail-event-attrs">
              {JSON.stringify((evt as { attributes?: unknown })?.attributes ?? {}, null, 2)}
            </pre>
          </div>
        ))}
      </Section>

      <Section
        label="Links"
        empty={(span.links ?? []).length === 0}
        emptyText="(none)"
      >
        {(span.links ?? []).map((link, idx) => (
          <div key={idx} className="span-detail-link">
            <span>{(link as { relationship?: string })?.relationship ?? "link"}</span>
            <span>→ {(link as { linked_span_id?: string })?.linked_span_id ?? "?"}</span>
          </div>
        ))}
      </Section>
    </div>
  );
}

function AttributeList({
  kind,
  attributes,
  duration_ms,
}: {
  kind: SpanNode["kind"];
  attributes: Record<string, unknown>;
  duration_ms: number | null | undefined;
}): ReactElement {
  const rows: Array<[string, string]> = [];
  const a = attributes;

  if (kind === "generation") {
    if (a.model_name) rows.push(["Model", String(a.model_name)]);
    if (a.provider) rows.push(["Provider", String(a.provider)]);
    if (a.prompt_tokens != null) rows.push(["Prompt tokens", formatInt(a.prompt_tokens)]);
    if (a.completion_tokens != null) rows.push(["Completion tokens", formatInt(a.completion_tokens)]);
    if (a.reasoning_tokens != null) rows.push(["Reasoning tokens", formatInt(a.reasoning_tokens)]);
    if (a.response_time_ms != null) rows.push(["Response time", `${a.response_time_ms}ms`]);
    if (a.finish_reason) rows.push(["Finish reason", String(a.finish_reason)]);
    if (a.has_thinking != null) rows.push(["Thinking", a.has_thinking ? "yes" : "no"]);
    if (a.has_tool_calls != null) rows.push(["Tool calls", a.has_tool_calls ? "yes" : "no"]);
  } else if (kind === "tool") {
    if (a.tool_name) rows.push(["Tool", String(a.tool_name)]);
    if (a.agent_name) rows.push(["Agent", String(a.agent_name)]);
    if (a.arguments != null) {
      rows.push(["Arguments", typeof a.arguments === "string" ? a.arguments : JSON.stringify(a.arguments)]);
    }
    if (a.result_summary) rows.push(["Result", String(a.result_summary)]);
  } else if (kind === "step") {
    if (a.agent_name) rows.push(["Agent", String(a.agent_name)]);
    if (a.step_number != null) rows.push(["Step #", String(a.step_number)]);
    if (a.action_type) rows.push(["Action", String(a.action_type)]);
    if (a.next_agents) rows.push(["Next", String(a.next_agents)]);
    if (a.success != null) rows.push(["Success", a.success ? "yes" : "no"]);
  } else if (kind === "branch") {
    if (a.branch_name) rows.push(["Branch", String(a.branch_name)]);
    if (a.source_agent) rows.push(["Source", String(a.source_agent)]);
    if (a.target_agents) rows.push(["Targets", String(a.target_agents)]);
    if (a.trigger_type) rows.push(["Trigger", String(a.trigger_type)]);
  } else if (kind === "execution") {
    if (a.task_summary) rows.push(["Task", String(a.task_summary)]);
    if (a.topology_summary) rows.push(["Topology", String(a.topology_summary)]);
    if (a.success != null) rows.push(["Success", a.success ? "yes" : "no"]);
    if (a.total_steps != null) rows.push(["Steps", String(a.total_steps)]);
  }
  if (duration_ms != null) rows.push(["Duration", `${duration_ms.toFixed(0)}ms`]);

  return (
    <dl className="attribute-list" data-testid="attribute-list">
      {rows.map(([label, value]) => (
        <div key={label} className="attribute-row">
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function ContentExpansions({
  kind,
  attributes,
}: {
  kind: SpanNode["kind"];
  attributes: Record<string, unknown>;
}): ReactElement | null {
  // Plan §10.2 + AC-364: only generation spans carry prompt/response
  // content. When content is captured (framework default
  // include_message_content=True), the toggles render an inline
  // expansion. When absent, the toggles render disabled with a
  // tooltip explaining the gap.
  if (kind !== "generation") return null;

  const prompt = attributes.prompt_content ?? attributes.prompt ?? null;
  const response = attributes.response_content ?? attributes.response ?? null;

  return (
    <div className="span-detail-content-section">
      <ContentToggle
        label="Show full prompt"
        content={prompt == null ? null : String(prompt)}
        testId="show-prompt"
      />
      <ContentToggle
        label="Show full response"
        content={response == null ? null : String(response)}
        testId="show-response"
      />
    </div>
  );
}

function ContentToggle({
  label,
  content,
  testId,
}: {
  label: string;
  content: string | null;
  testId: string;
}): ReactElement {
  const [open, setOpen] = useState(false);
  const disabled = content == null;
  return (
    <div className="content-toggle">
      <button
        type="button"
        className={`content-toggle-button${disabled ? " is-disabled" : ""}`}
        onClick={() => !disabled && setOpen((v) => !v)}
        data-testid={testId}
        aria-expanded={disabled ? undefined : open}
        disabled={disabled}
        title={disabled ? "Content not captured for this span" : undefined}
      >
        {disabled ? label : open ? `Hide ${label.replace(/^Show full /, "")}` : label}
        <span className="content-toggle-caret">{disabled ? "—" : open ? "▴" : "▾"}</span>
      </button>
      {!disabled && open && (
        <pre className="content-toggle-body" data-testid={`${testId}-body`}>
          {content}
        </pre>
      )}
    </div>
  );
}

function Section({
  label,
  children,
  empty,
  emptyText,
  ...rest
}: {
  label: string;
  empty?: boolean;
  emptyText?: string;
  children: React.ReactNode;
} & React.HTMLAttributes<HTMLDivElement>): ReactElement {
  return (
    <div className="span-detail-section" {...rest}>
      <div className="span-detail-section-label">{label}</div>
      <div className="span-detail-section-body">
        {empty ? <span className="span-detail-empty">{emptyText ?? "(none)"}</span> : children}
      </div>
    </div>
  );
}

function formatInt(n: unknown): string {
  if (typeof n !== "number") return String(n);
  return n.toLocaleString();
}
