/**
 * Typographic "tag" device — renders an inert `<element attrs... />`
 * markup snippet in the design system's mono face.
 *
 * Used three places in v0.3:
 *   - canvas empty state: `<agent name="" model="" tools={...} />`
 *   - workflow list empty state: `<workflow name="" agents={...} edges={...} />`
 *   - agent config form header: `<agent name="X" model="Y" tools={...} />`
 *
 * Attribute values can be strings (rendered in quotes) or lists
 * (rendered in braces, comma-joined — the "tools={a,b,c}" idiom).
 *
 * Size prop maps the three different font-sizes the three current
 * consumers use:
 *   - xs (11.5px): agent form header, inline within a flex row
 *   - sm (13px):    workflow list empty, balanced with body copy
 *   - md (14px):    canvas empty, prominent display
 */
import type { ReactElement } from "react";

import "./TagMarkup.css";

export type TagMarkupValue = string | readonly string[];
export type TagMarkupSize = "xs" | "sm" | "md";

export interface TagMarkupProps {
  /** The element name — "agent", "workflow", etc. */
  tag: string;
  /**
   * Attributes to render. Order preserved. String values become
   * `key="value"`; array values become `key={item1,item2}`.
   */
  attrs: ReadonlyArray<readonly [string, TagMarkupValue]>;
  size?: TagMarkupSize;
  /** When true, render inside a <pre> for multi-line display. */
  block?: boolean;
  className?: string;
}

export function TagMarkup({
  tag,
  attrs,
  size = "sm",
  block = false,
  className,
}: TagMarkupProps): ReactElement {
  const classes = [
    "ui-tagmarkup",
    `ui-tagmarkup--${size}`,
    block ? "ui-tagmarkup--block" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");

  const attrParts = attrs.map(([key, value]) => {
    if (Array.isArray(value)) {
      return `${key}={${value.join(",")}}`;
    }
    return `${key}="${value as string}"`;
  });

  const inner = block ? renderBlockBody(tag, attrParts) : renderInlineBody(tag, attrParts);

  if (block) {
    return <pre className={classes}>{inner}</pre>;
  }
  return <span className={classes}>{inner}</span>;
}

function renderInlineBody(tag: string, attrParts: string[]): string {
  return `<${tag}${attrParts.length > 0 ? " " + attrParts.join(" ") : ""} />`;
}

function renderBlockBody(tag: string, attrParts: string[]): string {
  if (attrParts.length === 0) return `<${tag} />`;
  // Indent continuation lines by the tag-name width + opening bracket.
  const indent = " ".repeat(tag.length + 2);
  const joined = attrParts.join("\n" + indent);
  return `<${tag} ${joined} />`;
}
